import os
import sys
import torch
import cv2
import numpy as np
from tqdm import tqdm
import json

_MULTIPOSE_DIR = os.path.dirname(os.path.abspath(__file__))
_MUSEPOSE_DIR = os.path.join(os.path.dirname(_MULTIPOSE_DIR), 'MusePose')
sys.path.insert(0, _MUSEPOSE_DIR)

from pose.script.dwpose import DWposeDetector, draw_pose

def generate_pose_video(video_path, output_path, detector, detect_resolution=512, image_resolution=720):
    """Generate a pose skeleton video from an input video."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing {video_path}: {total_frames} frames")

    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        # Detect pose
        pose_img, pose_dict = detector(
            frame,
            detect_resolution=detect_resolution,
            image_resolution=image_resolution,
            output_type='cv2',
            return_pose_dict=True
        )

        # Resize pose image to match original frame size
        pose_img_resized = cv2.resize(pose_img, (width, height))

        # Convert from RGB to BGR for OpenCV
        pose_img_bgr = cv2.cvtColor(pose_img_resized, cv2.COLOR_RGB2BGR)

        out.write(pose_img_bgr)

    cap.release()
    out.release()
    print(f"Saved pose video to {output_path}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize detector
    detector = DWposeDetector(
        det_config=os.path.join(_MUSEPOSE_DIR, "pose/config/yolox_l_8xb8-300e_coco.py"),
        det_ckpt=os.path.join(_MUSEPOSE_DIR, "pretrained_weights/dwpose/yolox_l_8x8_300e_coco.pth"),
        pose_config=os.path.join(_MUSEPOSE_DIR, "pose/config/dwpose-l_384x288.py"),
        pose_ckpt=os.path.join(_MUSEPOSE_DIR, "pretrained_weights/dwpose/dw-ll_ucoco_384.pth"),
        keypoints_only=False
    )
    detector = detector.to(device)

    # Video files to process
    video_dir = os.path.join(_MULTIPOSE_DIR, "assets/videos")
    pose_dir = os.path.join(_MULTIPOSE_DIR, "assets/videos_dwpose")
    os.makedirs(pose_dir, exist_ok=True)

    videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

    meta_info = []

    for video_file in videos:
        video_path = os.path.join(video_dir, video_file)
        pose_path = os.path.join(pose_dir, video_file)

        if not os.path.exists(pose_path):
            generate_pose_video(video_path, pose_path, detector)
        else:
            print(f"Pose video already exists: {pose_path}")

        # Verify frame counts match
        cap1 = cv2.VideoCapture(video_path)
        cap2 = cv2.VideoCapture(pose_path)
        frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        frames2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        cap1.release()
        cap2.release()

        if frames1 == frames2:
            meta_info.append({
                "video_path": os.path.abspath(video_path),
                "kps_path": os.path.abspath(pose_path)
            })
            print(f"Added {video_file} to training meta ({frames1} frames)")
        else:
            print(f"WARNING: Frame mismatch for {video_file}: {frames1} vs {frames2}")

    # Save meta info
    meta_path = os.path.join(_MULTIPOSE_DIR, "meta/training_meta.json")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, 'w') as f:
        json.dump(meta_info, f, indent=2)
    print(f"\nSaved training meta to {meta_path}")
    print(f"Total training videos: {len(meta_info)}")


if __name__ == "__main__":
    main()
