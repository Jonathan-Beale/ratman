import os
from PIL import Image
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MULTIPOSE_DIR = os.path.dirname(_SCRIPT_DIR)

# Directories
ORIG_DIR = os.path.join(_MULTIPOSE_DIR, 'assets/character_images')
POSE_DIR = os.path.join(_MULTIPOSE_DIR, 'output/ratman_pose_images')
OUT_DIR = os.path.join(_MULTIPOSE_DIR, 'output/ratman_pose_pairs')
os.makedirs(OUT_DIR, exist_ok=True)

# Get all image filenames (assume same names in both dirs)
filenames = [f for f in os.listdir(ORIG_DIR) if f.lower().endswith(('.png','.jpg','.jpeg'))]

for fname in tqdm(filenames):
    orig_path = os.path.join(ORIG_DIR, fname)
    pose_path = os.path.join(POSE_DIR, fname)
    if not os.path.exists(pose_path):
        continue
    orig_img = Image.open(orig_path).convert('RGB')
    pose_img = Image.open(pose_path).convert('RGB')
    # Create a new image side by side
    w, h = orig_img.size
    pair_img = Image.new('RGB', (w*2, h))
    pair_img.paste(orig_img, (0,0))
    pair_img.paste(pose_img, (w,0))
    pair_img.save(os.path.join(OUT_DIR, fname))
print('Paired images saved to', OUT_DIR)
