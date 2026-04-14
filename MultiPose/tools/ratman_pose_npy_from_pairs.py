
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
import numpy as np

# Directories
PAIR_DIR = os.path.join(_MULTIPOSE_DIR, 'output/ratman_pose_pairs_rerun')
ORIG_DIR = os.path.join(_MULTIPOSE_DIR, 'assets/character_images')
NPY_OUT_DIR = os.path.join(_MULTIPOSE_DIR, 'output/ratman_pose_npy')
os.makedirs(NPY_OUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detector = DWposeDetector(
    det_config = os.path.join(_MUSEPOSE_DIR, 'pose/config/yolox_l_8xb8-300e_coco.py'),
    det_ckpt = os.path.join(_MUSEPOSE_DIR, 'pretrained_weights/dwpose/yolox_l_8x8_300e_coco.pth'),
    pose_config = os.path.join(_MUSEPOSE_DIR, 'pose/config/dwpose-l_384x288.py'),
    pose_ckpt = os.path.join(_MUSEPOSE_DIR, 'pretrained_weights/dwpose/dw-ll_ucoco_384.pth'),
    keypoints_only=True
)
detector = detector.to(device)

filenames = [f for f in os.listdir(PAIR_DIR) if f.lower().endswith(('.png','.jpg','.jpeg'))]

for fname in tqdm(filenames):
    orig_path = os.path.join(ORIG_DIR, fname)
    npy_out_path = os.path.join(NPY_OUT_DIR, os.path.splitext(fname)[0] + '.npy')
    if not os.path.exists(orig_path):
        continue
    img = Image.open(orig_path).convert('RGB')
    pose = detector(img)
    np.save(npy_out_path, pose)
print('Pose .npy files saved to', NPY_OUT_DIR)
