"""
ControlNet + Stable Diffusion Pose-Guided Generation
Generates images based on pose data from pose estimation systems
"""

import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime

class PoseGuidedGenerator:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5",
                 controlnet_id="lllyasviel/control_v11p_sd15_openpose"):
        """
        Initialize the pose-guided image generator

        Args:
            model_id: Base Stable Diffusion model
            controlnet_id: ControlNet model for pose control
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        print(f"Loading ControlNet model: {controlnet_id}")
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )

        print(f"Loading Stable Diffusion pipeline: {model_id}")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_id,
            controlnet=self.controlnet,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None
        )
        self.pipe = self.pipe.to(self.device)

        # Use faster scheduler
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

        # Enable memory optimizations
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
            self.pipe.enable_model_cpu_offload()

        print("Pipeline loaded successfully!")

    def generate_from_pose_image(self, pose_image_path, prompt,
                                  output_dir="outputs/controlnet",
                                  num_inference_steps=30,
                                  guidance_scale=7.5,
                                  controlnet_conditioning_scale=1.0):
        """
        Generate an image based on a pose image

        Args:
            pose_image_path: Path to pose skeleton image (OpenPose format)
            prompt: Text prompt for image generation
            output_dir: Directory to save output
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            controlnet_conditioning_scale: How strongly to follow the pose

        Returns:
            Path to generated image
        """
        os.makedirs(output_dir, exist_ok=True)

        # Load pose image
        pose_image = load_image(pose_image_path)

        print(f"Generating image for prompt: '{prompt}'")
        print(f"Using pose from: {pose_image_path}")

        # Generate image
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                image=pose_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale
            )

        image = result.images[0]

        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_controlnet_{prompt[:30].replace(' ', '_')}.png"
        output_path = os.path.join(output_dir, filename)
        image.save(output_path)

        print(f"Image saved to: {output_path}")
        return output_path

    def create_openpose_image_from_keypoints(self, keypoints, image_shape=(512, 512)):
        """
        Create an OpenPose-style skeleton image from keypoint data
        Useful for integrating with existing pose estimation systems

        Args:
            keypoints: Array of keypoints (N, 3) where each row is [x, y, confidence]
                      Should follow COCO keypoint format
            image_shape: Output image shape (height, width)

        Returns:
            PIL Image with skeleton drawn
        """
        # COCO keypoint connections (skeleton)
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]

        # Create blank image
        canvas = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)

        # Draw connections
        for start_idx, end_idx in connections:
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                start_point = keypoints[start_idx]
                end_point = keypoints[end_idx]

                # Only draw if both points are visible (confidence > 0)
                if start_point[2] > 0 and end_point[2] > 0:
                    pt1 = (int(start_point[0]), int(start_point[1]))
                    pt2 = (int(end_point[0]), int(end_point[1]))
                    cv2.line(canvas, pt1, pt2, (255, 255, 255), 2)

        # Draw keypoints
        for keypoint in keypoints:
            if keypoint[2] > 0:  # If visible
                center = (int(keypoint[0]), int(keypoint[1]))
                cv2.circle(canvas, center, 4, (0, 255, 0), -1)

        return Image.fromarray(canvas)

    def generate_from_keypoints(self, keypoints, prompt,
                                output_dir="outputs/controlnet",
                                image_shape=(512, 512),
                                **kwargs):
        """
        Generate an image directly from keypoint data

        Args:
            keypoints: Array of keypoints (N, 3) where each row is [x, y, confidence]
            prompt: Text prompt for image generation
            output_dir: Directory to save output
            image_shape: Shape for pose image
            **kwargs: Additional arguments passed to generate_from_pose_image

        Returns:
            Path to generated image
        """
        # Create pose image from keypoints
        pose_image = self.create_openpose_image_from_keypoints(keypoints, image_shape)

        # Save temporary pose image
        os.makedirs(output_dir, exist_ok=True)
        temp_pose_path = os.path.join(output_dir, "temp_pose.png")
        pose_image.save(temp_pose_path)

        # Generate image using pose
        result = self.generate_from_pose_image(temp_pose_path, prompt, output_dir, **kwargs)

        return result


def demo_with_sample_pose():
    """
    Demo function that generates an image with a sample pose
    """
    # Initialize generator
    generator = PoseGuidedGenerator()

    # Create a simple standing pose (approximate keypoints)
    # Format: [x, y, confidence] for each keypoint
    # COCO format: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
    sample_keypoints = np.array([
        [256, 100, 1.0],  # 0: nose
        [246, 90, 1.0],   # 1: left eye
        [266, 90, 1.0],   # 2: right eye
        [236, 100, 1.0],  # 3: left ear
        [276, 100, 1.0],  # 4: right ear
        [220, 180, 1.0],  # 5: left shoulder
        [292, 180, 1.0],  # 6: right shoulder
        [200, 260, 1.0],  # 7: left elbow
        [312, 260, 1.0],  # 8: right elbow
        [190, 340, 1.0],  # 9: left wrist
        [322, 340, 1.0],  # 10: right wrist
        [230, 300, 1.0],  # 11: left hip
        [282, 300, 1.0],  # 12: right hip
        [230, 400, 1.0],  # 13: left knee
        [282, 400, 1.0],  # 14: right knee
        [230, 500, 1.0],  # 15: left ankle
        [282, 500, 1.0],  # 16: right ankle
    ])

    # Generate image with this pose
    prompt = "a superhero in a dynamic pose, detailed costume, cinematic lighting, 4k"
    print("\n" + "="*60)
    print("DEMO: Generating image from sample standing pose")
    print("="*60)

    output_path = generator.generate_from_keypoints(
        keypoints=sample_keypoints,
        prompt=prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        controlnet_conditioning_scale=1.0
    )

    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print(f"Generated image: {output_path}")
    print("="*60)


if __name__ == "__main__":
    demo_with_sample_pose()
