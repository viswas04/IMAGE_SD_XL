## SDXL Text-to-Image (Diffusers)

Generate high-quality, realistic images from text prompts using Stable Diffusion XL (SDXL) via the Hugging Face Diffusers library.

### 1) Requirements

- Python 3.10+
- A CUDA-compatible GPU with at least 8-12GB VRAM recommended for 1024×1024. CPU works but will be very slow.

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install PyTorch separately with the correct CUDA version (see the official selector):

```bash
# Example for CUDA 12.x (update if needed):
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

You may need to accept the SDXL model license on Hugging Face for the first download.

### 2) Usage

Basic generation (1024×1024):

```bash
python generate_sdxl.py --prompt "a hyper-realistic photo of a red fox in a misty forest at dawn" --output outputs/fox.png
```

With refiner for extra detail:

```bash
python generate_sdxl.py --prompt "an elegant modern living room, soft natural light, 35mm" --use_refiner --output outputs/living_room.png
```

Extra options:

- `--negative_prompt`: steer away from artifacts, e.g. "blurry, low quality, text, watermark"
- `--height` / `--width`: multiples of 8; 1024×1024 recommended
- `--steps`: denoising steps (e.g. 30–50)
- `--guidance_scale`: CFG scale (e.g. 5–8)
- `--seed`: set for reproducibility
- `--refiner_start`: fraction of steps (0–1) where refinement begins (default 0.8)
- `--cpu`: force CPU (very slow)
- `--no_offload`: disable CPU offloading on CUDA

