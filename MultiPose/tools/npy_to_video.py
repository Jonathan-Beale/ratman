import os
import cv2
import numpy as np
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MULTIPOSE_DIR = os.path.dirname(_SCRIPT_DIR)
_MUSEPOSE_DIR = os.path.join(os.path.dirname(_MULTIPOSE_DIR), 'MusePose')
sys.path.insert(0, _MUSEPOSE_DIR)

from pose.script.dwpose import draw_pose

POSE_FILES = [
    os.path.join(_MUSEPOSE_DIR, 'output/dwpose_keypoints/VID_20260203_154431864.npy'),
    os.path.join(_MUSEPOSE_DIR, 'output/dwpose_keypoints/VID_20260203_154444045.npy'),
]
OUTPUT_DIR = os.path.join(_MUSEPOSE_DIR, 'output/pose_videos')
WIDTH, HEIGHT = 384, 384
FPS = 24

def npy_to_video(npy_path, out_path):
    poses = np.load(npy_path, allow_pickle=True)
    num_frames = poses.shape[0]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(out_path, fourcc, FPS, (WIDTH, HEIGHT))
    for i in range(num_frames):
        frame = poses[i]
        img = draw_pose(frame, HEIGHT, WIDTH, draw_face=False)
        # Rotate image 90 degrees counterclockwise (upright)
        img_rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        video.write(img_rotated)
    video.release()

os.makedirs(OUTPUT_DIR, exist_ok=True)
for pose_file in POSE_FILES:
    base = os.path.splitext(os.path.basename(pose_file))[0]
    out_path = os.path.join(OUTPUT_DIR, base + '.mp4')
    npy_to_video(pose_file, out_path)
print('Pose videos generated.')
