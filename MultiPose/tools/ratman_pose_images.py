
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MULTIPOSE_DIR = os.path.dirname(_SCRIPT_DIR)
_MUSEPOSE_DIR = os.path.join(os.path.dirname(_MULTIPOSE_DIR), 'MusePose')
sys.path.insert(0, _MUSEPOSE_DIR)

from tqdm import tqdm
from PIL import Image
from pose.script.dwpose import DWposeDetector, draw_pose
import torch
import cv2

# Directory containing ratman images
INPUT_DIR = os.path.join(_MULTIPOSE_DIR, 'assets/character_images')
OUTPUT_DIR = os.path.join(_MULTIPOSE_DIR, 'output/ratman_pose_images')

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detector = DWposeDetector(
    det_config = os.path.join(_MUSEPOSE_DIR, 'pose/config/yolox_l_8xb8-300e_coco.py'),
    det_ckpt = os.path.join(_MUSEPOSE_DIR, 'pretrained_weights/dwpose/yolox_l_8x8_300e_coco.pth'),
    pose_config = os.path.join(_MUSEPOSE_DIR, 'pose/config/dwpose-l_384x288.py'),
    pose_ckpt = os.path.join(_MUSEPOSE_DIR, 'pretrained_weights/dwpose/dw-ll_ucoco_384.pth'),
    keypoints_only=False
)
detector = detector.to(device)

for fname in tqdm(os.listdir(INPUT_DIR)):
    if not (fname.endswith('.png') or fname.endswith('.jpg') or fname.endswith('.jpeg')):
        continue
    in_path = os.path.join(INPUT_DIR, fname)
    out_path = os.path.join(OUTPUT_DIR, fname)
    img = Image.open(in_path).convert('RGB')
    detected_map, _ = detector(img)
    if isinstance(detected_map, Image.Image):
        detected_map.save(out_path)
    else:
        cv2.imwrite(out_path, detected_map)
print('Pose estimation images saved to', OUTPUT_DIR)
