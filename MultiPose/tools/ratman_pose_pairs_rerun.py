
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MULTIPOSE_DIR = os.path.dirname(_SCRIPT_DIR)
_MUSEPOSE_DIR = os.path.join(os.path.dirname(_MULTIPOSE_DIR), 'MusePose')
sys.path.insert(0, _MUSEPOSE_DIR)

from PIL import Image
from tqdm import tqdm
from pose.script.dwpose import DWposeDetector
import torch
import cv2

# Directories
PAIR_DIR = os.path.join(_MULTIPOSE_DIR, 'output/ratman_pose_pairs')
ORIG_DIR = os.path.join(_MULTIPOSE_DIR, 'assets/character_images')
POSE_DIR = os.path.join(_MULTIPOSE_DIR, 'output/ratman_pose_images')
OUT_DIR = os.path.join(_MULTIPOSE_DIR, 'output/ratman_pose_pairs_rerun')
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detector = DWposeDetector(
    det_config = os.path.join(_MUSEPOSE_DIR, 'pose/config/yolox_l_8xb8-300e_coco.py'),
    det_ckpt = os.path.join(_MUSEPOSE_DIR, 'pretrained_weights/dwpose/yolox_l_8x8_300e_coco.pth'),
    pose_config = os.path.join(_MUSEPOSE_DIR, 'pose/config/dwpose-l_384x288.py'),
    pose_ckpt = os.path.join(_MUSEPOSE_DIR, 'pretrained_weights/dwpose/dw-ll_ucoco_384.pth'),
    keypoints_only=False
)
detector = detector.to(device)

filenames = [f for f in os.listdir(PAIR_DIR) if f.lower().endswith(('.png','.jpg','.jpeg'))]

for fname in tqdm(filenames):
    orig_path = os.path.join(ORIG_DIR, fname)
    out_pose_path = os.path.join(POSE_DIR, fname)
    out_pair_path = os.path.join(OUT_DIR, fname)
    if not os.path.exists(orig_path):
        continue
    img = Image.open(orig_path).convert('RGB')
    detected_map, _ = detector(img)
    if isinstance(detected_map, Image.Image):
        detected_map.save(out_pose_path)
    else:
        cv2.imwrite(out_pose_path, detected_map)
    # Pair
    w, h = img.size
    pose_img = Image.open(out_pose_path).convert('RGB')
    pair_img = Image.new('RGB', (w*2, h))
    pair_img.paste(img, (0,0))
    pair_img.paste(pose_img, (w,0))
    pair_img.save(out_pair_path)
print('Rerun paired images saved to', OUT_DIR)
