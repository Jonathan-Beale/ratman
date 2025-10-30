"""
Complete Pose-to-Video Pipeline
Integrates pose estimation with ControlNet and AnimateDiff for temporal generation
"""

import torch
import numpy as np
from PIL import Image
import cv2
import os
from datetime import datetime
from pathlib import Path

try:
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
    from diffusers.utils import export_to_video
except ImportError:
    print("Some dependencies not installed. Install with: pip install diffusers[torch]")


class PoseToVideoGenerator:
    """
    Complete pipeline for generating videos from pose sequences
    """

    def __init__(self, use_controlnet=True):
        """
        Initialize the pose-to-video generator

        Args:
            use_controlnet: Whether to use ControlNet for pose guidance
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.use_controlnet = use_controlnet

        if use_controlnet:
            self._init_controlnet_pipeline()

    def _init_controlnet_pipeline(self):
        """Initialize ControlNet pipeline for pose-guided generation"""
        print("Initializing ControlNet pipeline...")

        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_openpose",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )

        self.control_pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None
        )
        self.control_pipe = self.control_pipe.to(self.device)

        if self.device == "cuda":
            self.control_pipe.enable_attention_slicing()

        print("ControlNet pipeline ready!")

    def _init_animatediff_pipeline(self):
        """Initialize AnimateDiff pipeline for video generation"""
        print("Initializing AnimateDiff pipeline...")

        # Load motion adapter
        adapter = MotionAdapter.from_pretrained(
            "guoyww/animatediff-motion-adapter-v1-5-2",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )

        self.video_pipe = AnimateDiffPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            motion_adapter=adapter,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.video_pipe.scheduler = DDIMScheduler.from_config(
            self.video_pipe.scheduler.config,
            beta_schedule="linear",
            clip_sample=False
        )
        self.video_pipe = self.video_pipe.to(self.device)

        if self.device == "cuda":
            self.video_pipe.enable_attention_slicing()
            self.video_pipe.enable_vae_slicing()

        print("AnimateDiff pipeline ready!")

    def create_pose_image_from_keypoints(self, keypoints, image_shape=(512, 512)):
        """
        Create OpenPose skeleton image from keypoints

        Args:
            keypoints: Array (N, 3) of [x, y, confidence]
            image_shape: Output image shape

        Returns:
            PIL Image with skeleton
        """
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]

        canvas = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)

        # Draw connections
        for start_idx, end_idx in connections:
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                start_point = keypoints[start_idx]
                end_point = keypoints[end_idx]

                if start_point[2] > 0 and end_point[2] > 0:
                    pt1 = (int(start_point[0]), int(start_point[1]))
                    pt2 = (int(end_point[0]), int(end_point[1]))
                    cv2.line(canvas, pt1, pt2, (255, 255, 255), 2)

        # Draw keypoints
        for keypoint in keypoints:
            if keypoint[2] > 0:
                center = (int(keypoint[0]), int(keypoint[1]))
                cv2.circle(canvas, center, 4, (0, 255, 0), -1)

        return Image.fromarray(canvas)

    def generate_image_from_pose(self, keypoints, prompt,
                                 output_dir="outputs/pose_to_video",
                                 **kwargs):
        """
        Generate a single image from pose keypoints

        Args:
            keypoints: Pose keypoints array
            prompt: Text prompt
            output_dir: Output directory
            **kwargs: Additional generation parameters

        Returns:
            PIL Image
        """
        if not self.use_controlnet:
            raise ValueError("ControlNet not initialized. Set use_controlnet=True")

        # Create pose image
        pose_image = self.create_pose_image_from_keypoints(keypoints)

        # Generate with ControlNet
        with torch.no_grad():
            result = self.control_pipe(
                prompt=prompt,
                image=pose_image,
                num_inference_steps=kwargs.get('num_inference_steps', 20),
                guidance_scale=kwargs.get('guidance_scale', 7.5),
                controlnet_conditioning_scale=kwargs.get('controlnet_scale', 1.0)
            )

        return result.images[0]

    def generate_video_from_pose_sequence(self, pose_sequence, prompt,
                                         output_dir="outputs/pose_to_video",
                                         output_filename=None):
        """
        Generate video from a sequence of poses

        Args:
            pose_sequence: List of keypoint arrays, one per frame
            prompt: Text prompt for video generation
            output_dir: Output directory
            output_filename: Custom output filename

        Returns:
            Path to generated video
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Generating video from {len(pose_sequence)} poses")
        print(f"Prompt: '{prompt}'")
        print(f"{'='*60}")

        # Generate image for each pose
        frames = []
        for i, keypoints in enumerate(pose_sequence):
            print(f"Processing pose {i+1}/{len(pose_sequence)}...")
            frame = self.generate_image_from_pose(keypoints, prompt)
            frames.append(frame)

        # Save as video
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{timestamp}_pose_video.mp4"

        output_path = os.path.join(output_dir, output_filename)

        # Convert frames to video using opencv
        if frames:
            height, width = frames[0].size[1], frames[0].size[0]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, 10.0, (width, height))

            for frame in frames:
                # Convert PIL to opencv format
                frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                video_writer.write(frame_cv)

            video_writer.release()

        print(f"\nVideo saved to: {output_path}")
        return output_path

    def generate_video_with_animatediff(self, prompt,
                                       output_dir="outputs/animatediff",
                                       num_frames=16,
                                       num_inference_steps=25):
        """
        Generate video using AnimateDiff (text-to-video)

        Args:
            prompt: Text prompt
            output_dir: Output directory
            num_frames: Number of frames to generate
            num_inference_steps: Inference steps

        Returns:
            Path to generated video
        """
        if not hasattr(self, 'video_pipe'):
            self._init_animatediff_pipeline()

        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Generating {num_frames}-frame video with AnimateDiff")
        print(f"Prompt: '{prompt}'")
        print(f"{'='*60}")

        with torch.no_grad():
            result = self.video_pipe(
                prompt=prompt,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=7.5
            )

        frames = result.frames[0]

        # Save video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"{timestamp}_animatediff.gif")
        export_to_video(frames, output_path, fps=8)

        print(f"Video saved to: {output_path}")
        return output_path


def demo_pose_sequence():
    """
    Demo: Generate a simple animation from pose sequence
    """
    print("\n" + "="*60)
    print("POSE-TO-VIDEO PIPELINE DEMO")
    print("="*60)

    # Create a simple pose sequence (person raising arms)
    base_pose = np.array([
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

    # Create sequence of 5 poses with arms gradually raising
    pose_sequence = []
    for i in range(5):
        pose = base_pose.copy()
        # Animate left wrist (idx 9) and right wrist (idx 10) moving up
        pose[9][1] = 340 - (i * 40)  # left wrist moves up
        pose[10][1] = 340 - (i * 40)  # right wrist moves up
        # Adjust elbows accordingly
        pose[7][1] = 260 - (i * 20)  # left elbow
        pose[8][1] = 260 - (i * 20)  # right elbow
        pose_sequence.append(pose)

    # Generate video from poses
    generator = PoseToVideoGenerator(use_controlnet=True)
    prompt = "a person in athletic wear, gym background, professional lighting"

    output_path = generator.generate_video_from_pose_sequence(
        pose_sequence=pose_sequence,
        prompt=prompt
    )

    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print(f"Video saved to: {output_path}")
    print("="*60)


if __name__ == "__main__":
    demo_pose_sequence()
