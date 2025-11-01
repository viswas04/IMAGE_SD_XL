import argparse
import os
import random
import torch


def create_base_pipeline(device: str, dtype: torch.dtype, enable_offload: bool):
	# Lazy import heavy ML libraries so the module can be imported for quick dry-runs
	from diffusers import DiffusionPipeline, AutoencoderKL

	pipeline = DiffusionPipeline.from_pretrained(
		"stabilityai/stable-diffusion-xl-base-1.0",
		variant="fp16" if dtype == torch.float16 else None,
		torch_dtype=dtype,
		use_safetensors=True,
	)
	# Optional: use a faster VAE for decoding if available
	try:
		vae = AutoencoderKL.from_pretrained(
			"madebyollin/sdxl-vae-fp16-fix" if dtype == torch.float16 else "stabilityai/sdxl-vae",
			torch_dtype=dtype,
			use_safetensors=True,
		)
		pipeline.vae = vae
	except Exception:
		pass

	if enable_offload and device == "cuda":
		pipeline.enable_model_cpu_offload()
	else:
		pipeline.to(device)

	# Speed/VRAM tradeoffs
	if device == "cuda":
		pipeline.enable_vae_tiling()
		pipeline.enable_xformers_memory_efficient_attention()

	return pipeline


def create_refiner_pipeline(device: str, dtype: torch.dtype, enable_offload: bool):
	# Lazy import to avoid requiring `diffusers` on simple help/dry-run usages
	from diffusers import DiffusionPipeline

	refiner = DiffusionPipeline.from_pretrained(
		"stabilityai/stable-diffusion-xl-refiner-1.0",
		variant="fp16" if dtype == torch.float16 else None,
		torch_dtype=dtype,
		use_safetensors=True,
	)
	if enable_offload and device == "cuda":
		refiner.enable_model_cpu_offload()
	else:
		refiner.to(device)
	refiner.enable_vae_tiling()
	return refiner


def parse_args():
	parser = argparse.ArgumentParser(description="Generate an image with SDXL from a text prompt.")
	parser.add_argument("--prompt", type=str, required=True, help="Text prompt describing the image.")
	parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt to avoid undesired elements.")
	parser.add_argument("--output", type=str, default="output.png", help="Output image file path.")
	parser.add_argument("--height", type=int, default=864, help="Output image height (multiple of 8, default 1536x864 16:9).")
	parser.add_argument("--width", type=int, default=1536, help="Output image width (multiple of 8, default 1536x864 16:9).")
	parser.add_argument("--aspect", type=str, default=None, help="Optional aspect ratio, e.g. 16:9 or 4:3. If provided, height/width will be computed if missing.")
	parser.add_argument("--preset_16_9", action="store_true", help="Shortcut to use a sensible 16:9 size (1536x864) when no width/height specified.")
	parser.add_argument("--steps", type=int, default=30, help="Number of denoising steps.")
	parser.add_argument("--guidance_scale", type=float, default=7.0, help="Classifier-free guidance scale.")
	parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
	parser.add_argument("--use_refiner", action="store_true", help="Use SDXL refiner for final denoising.")
	parser.add_argument("--refiner_start", type=float, default=0.8, help="Fraction of steps after which to hand off to refiner (0-1).")
	parser.add_argument("--cpu", action="store_true", help="Force CPU mode (very slow for SDXL).")
	parser.add_argument("--no_offload", action="store_true", help="Disable CPU offload even on CUDA.")
	parser.add_argument("--dry_run", action="store_true", help="Validate args and exit without loading models (no diffusers import required).")
	return parser.parse_args()


def main():
	
	args = parse_args()

	device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
	# fp16 only on CUDA. CPU uses float32.
	dtype = torch.float16 if device == "cuda" else torch.float32
	enable_offload = (not args.no_offload)

	if args.seed is None:
		args.seed = random.randint(0, 2**31 - 1)
	rng = torch.Generator(device=device).manual_seed(args.seed)

	print(f"Device: {device} | DType: {dtype} | Seed: {args.seed}")

	# Quick exit mode for testing argument parsing and environment without heavy deps
	if args.dry_run:
		print("Dry run: exiting before loading models. No diffusers/transformers required.")
		return

	# Helper: apply aspect ratio or presets and ensure dimensions are multiples of 8
	def _round8(x: int) -> int:
		return max(8, (int(x) // 8) * 8)

	# Apply 16:9 preset if requested and user didn't manually set dimensions
	if args.preset_16_9 and (args.width == 1024 and args.height == 1024):
		args.width = 1536
		args.height = 864

	if args.aspect:
		try:
			aw, ah = args.aspect.split(":")
			aw = float(aw)
			ah = float(ah)
			if args.width and not args.height:
				args.height = _round8(args.width * (ah / aw))
			elif args.height and not args.width:
				args.width = _round8(args.height * (aw / ah))
			elif not args.width and not args.height:
				# default to a reasonably large width
				args.width = 1536
				args.height = _round8(args.width * (ah / aw))
			else:
				# both set: leave as-is but ensure multiples of 8
				args.width = _round8(args.width)
				args.height = _round8(args.height)
		except Exception:
			print(f"Warning: invalid --aspect value '{args.aspect}'. Expected format like 16:9. Ignoring.")
			args.width = _round8(args.width)
			args.height = _round8(args.height)
	else:
		# Ensure multiples of 8 for safety
		args.width = _round8(args.width)
		args.height = _round8(args.height)

	print("Loading SDXL base pipeline…")
	pipe = create_base_pipeline(device, dtype, enable_offload)

	# Base generation (optionally with refiner handoff)
	if args.use_refiner:
		print("Loading SDXL refiner…")
		refiner = create_refiner_pipeline(device, dtype, enable_offload)

		# First pass with base, stopping at refiner_start
		image_latents = pipe(
			prompt=args.prompt,
			negative_prompt=args.negative_prompt if args.negative_prompt else None,
			height=args.height,
			width=args.width,
			num_inference_steps=args.steps,
			guidance_scale=args.guidance_scale,
			output_type="latent",
			generator=rng,
			denoising_end=args.refiner_start,
		).images

		# Refine the remaining steps
		image = refiner(
			prompt=args.prompt,
			negative_prompt=args.negative_prompt if args.negative_prompt else None,
			num_inference_steps=args.steps,
			guidance_scale=args.guidance_scale,
			generator=rng,
			image=image_latents,
			denoising_start=args.refiner_start,
		).images[0]
	else:
		image = pipe(
			prompt=args.prompt,
			negative_prompt=args.negative_prompt if args.negative_prompt else None,
			height=args.height,
			width=args.width,
			num_inference_steps=args.steps,
			guidance_scale=args.guidance_scale,
			generator=rng,
		).images[0]

	# Ensure output directory exists
	os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
	image.save(args.output)
	print(f"Saved: {args.output}")


if __name__ == "__main__":
	main()


