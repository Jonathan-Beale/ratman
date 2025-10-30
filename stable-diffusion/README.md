# Stable Diffusion Image Generation

This subfolder contains a Stable Diffusion implementation for generating images from text prompts.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the image generation script:
```bash
python generate_image.py
```

By default, this will generate an image of Batman and save it to the `outputs/` directory.

## Customization

Edit `generate_image.py` to change the prompt or modify generation parameters like:
- `num_inference_steps`: Number of denoising steps (higher = better quality but slower)
- `guidance_scale`: How closely to follow the prompt (7.5 is standard)
- `model_id`: Different Stable Diffusion model variants

## Requirements

- Python 3.8+
- PyTorch
- diffusers library
- transformers library
- GPU recommended but not required (will use CPU if no GPU available)
