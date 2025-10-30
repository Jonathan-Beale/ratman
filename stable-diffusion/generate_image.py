"""
Stable Diffusion Image Generation Script
Generates images using the Stable Diffusion model
"""

import torch
from diffusers import StableDiffusionPipeline
import os
from datetime import datetime

def generate_image(prompt, output_dir="outputs", model_id="runwayml/stable-diffusion-v1-5"):
    """
    Generate an image using Stable Diffusion

    Args:
        prompt (str): Text prompt for image generation
        output_dir (str): Directory to save generated images
        model_id (str): Hugging Face model ID

    Returns:
        str: Path to the generated image
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the model
    print(f"Loading Stable Diffusion model: {model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None  # Disable safety checker for faster inference
    )
    pipe = pipe.to(device)

    # Enable memory efficient attention if using CUDA
    if device == "cuda":
        pipe.enable_attention_slicing()

    print(f"Generating image for prompt: '{prompt}'")

    # Generate image
    with torch.no_grad():
        image = pipe(
            prompt,
            num_inference_steps=50,
            guidance_scale=7.5
        ).images[0]

    # Save image with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{prompt[:30].replace(' ', '_')}.png"
    output_path = os.path.join(output_dir, filename)
    image.save(output_path)

    print(f"Image saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    # Generate Batman image
    prompt = "Batman, dark knight, cinematic lighting, highly detailed, digital art, 4k"
    output_path = generate_image(prompt)
    print(f"Successfully generated image: {output_path}")
