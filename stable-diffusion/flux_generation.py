"""
FLUX Image Generation
Using FLUX model from Black Forest Labs for high-quality image generation
"""

import torch
from diffusers import FluxPipeline
from PIL import Image
import os
from datetime import datetime

class FluxGenerator:
    def __init__(self, model_id="black-forest-labs/FLUX.1-schnell"):
        """
        Initialize FLUX generator

        Available models:
        - "black-forest-labs/FLUX.1-schnell": Fast variant (4-step generation)
        - "black-forest-labs/FLUX.1-dev": Higher quality (requires more steps)

        Args:
            model_id: FLUX model to use
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        if self.device == "cpu":
            print("WARNING: FLUX is designed for GPU. CPU inference will be very slow.")

        print(f"Loading FLUX model: {model_id}")
        print("Note: FLUX models are large (~24GB). This may take a while...")

        try:
            self.pipe = FluxPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
            )
            self.pipe = self.pipe.to(self.device)

            # Memory optimizations for GPU
            if self.device == "cuda":
                self.pipe.enable_attention_slicing()
                self.pipe.vae.enable_slicing()
                self.pipe.vae.enable_tiling()

            self.model_id = model_id
            print("FLUX pipeline loaded successfully!")

        except Exception as e:
            print(f"Error loading FLUX model: {e}")
            print("\nTroubleshooting:")
            print("1. FLUX requires large GPU memory (24GB+ recommended)")
            print("2. You may need to accept the model license on HuggingFace")
            print("3. Use FLUX.1-schnell for faster/lighter inference")
            raise

    def generate(self, prompt, output_dir="outputs/flux",
                 height=1024, width=1024,
                 num_inference_steps=4,
                 guidance_scale=0.0,
                 num_images=1):
        """
        Generate images with FLUX

        Args:
            prompt: Text prompt for generation
            output_dir: Directory to save outputs
            height: Image height (FLUX works best at 1024)
            width: Image width (FLUX works best at 1024)
            num_inference_steps: Number of steps (4 for schnell, 20-50 for dev)
            guidance_scale: CFG scale (0.0 for schnell, 3.5-7.5 for dev)
            num_images: Number of images to generate

        Returns:
            List of paths to generated images
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nGenerating {num_images} image(s) with FLUX")
        print(f"Prompt: '{prompt}'")
        print(f"Resolution: {width}x{height}")
        print(f"Steps: {num_inference_steps}")

        # Adjust parameters based on model variant
        if "schnell" in self.model_id.lower():
            if num_inference_steps > 4:
                print(f"Note: FLUX.1-schnell is optimized for 4 steps. Using {num_inference_steps} steps.")
            if guidance_scale != 0.0:
                print(f"Note: FLUX.1-schnell works best with guidance_scale=0.0")

        output_paths = []

        with torch.no_grad():
            for i in range(num_images):
                print(f"\nGenerating image {i+1}/{num_images}...")

                # Generate image
                result = self.pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=torch.Generator(device=self.device).manual_seed(42 + i)
                )

                image = result.images[0]

                # Save image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{timestamp}_flux_{i+1}_{prompt[:30].replace(' ', '_')}.png"
                output_path = os.path.join(output_dir, filename)
                image.save(output_path)

                print(f"Image saved to: {output_path}")
                output_paths.append(output_path)

        return output_paths

    def generate_variations(self, prompt, num_variations=3,
                           output_dir="outputs/flux", **kwargs):
        """
        Generate multiple variations of the same prompt

        Args:
            prompt: Text prompt
            num_variations: Number of variations to generate
            output_dir: Output directory
            **kwargs: Additional arguments for generate()

        Returns:
            List of paths to generated images
        """
        print(f"\n{'='*60}")
        print(f"Generating {num_variations} variations")
        print(f"{'='*60}")

        return self.generate(prompt, output_dir, num_images=num_variations, **kwargs)


class FluxControlGenerator:
    """
    FLUX with ControlNet-style conditioning
    Note: Official FLUX ControlNet may not be available yet,
    this is a placeholder for future implementation
    """
    def __init__(self):
        print("Note: FLUX ControlNet integration is currently experimental.")
        print("For pose control, use the ControlNet + Stable Diffusion implementation.")
        raise NotImplementedError(
            "FLUX ControlNet is not yet officially released. "
            "Use controlnet_pose_generation.py for pose-guided generation."
        )


def demo_flux_generation():
    """
    Demo function for FLUX generation
    """
    print("\n" + "="*60)
    print("FLUX IMAGE GENERATION DEMO")
    print("="*60)

    try:
        # Initialize FLUX generator
        # Using schnell variant for faster generation
        generator = FluxGenerator(model_id="black-forest-labs/FLUX.1-schnell")

        # Generate a high-quality image
        prompt = "Batman standing on a rooftop at night, Gotham city skyline, dramatic lighting, cinematic, highly detailed, 8k"

        print("\n" + "="*60)
        print("Generating Batman image with FLUX...")
        print("="*60)

        output_paths = generator.generate(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=4,  # Optimal for schnell variant
            guidance_scale=0.0,     # Schnell uses distillation, no CFG needed
            num_images=1
        )

        print("\n" + "="*60)
        print("FLUX DEMO COMPLETE!")
        print(f"Generated images: {output_paths}")
        print("="*60)

    except Exception as e:
        print(f"\n{'='*60}")
        print("FLUX Demo Failed")
        print(f"{'='*60}")
        print(f"Error: {e}")
        print("\nThis might be because:")
        print("1. FLUX requires significant GPU memory (16GB+ VRAM)")
        print("2. Model license acceptance on HuggingFace may be required")
        print("3. Network issues during model download")
        print("\nFalling back to Stable Diffusion would be recommended for systems with less VRAM.")


if __name__ == "__main__":
    demo_flux_generation()
