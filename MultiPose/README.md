# MultiPose

Fine-tuning pipeline for MusePose using synthetic multi-pose training data. Generates DWPose skeleton images from character reference images and prepares training pairs.

## Overview

```
assets/
  character_images/     reference images of the character
output/                 generated pose images, pairs, and npy files
tools/                  pose generation and dataset preparation scripts
MusePose/               MusePose fork with multi-actor configs and training scripts
```

The workflow takes character reference images, runs DWPose detection to generate skeleton images, assembles training pairs, and launches MusePose stage-1 fine-tuning.

## Usage

### 1. Add character images

Place reference images (`.png` / `.jpg`) in `assets/character_images/`.

### 2. Generate pose images

Runs DWPose detection on each character image and saves the skeleton overlay:

```bash
python3 tools/ratman_pose_images.py
```

Output: `output/ratman_pose_images/`

### 3. Create training pairs

Assembles side-by-side (reference image, pose image) pairs:

```bash
python3 tools/ratman_pose_pairs.py
```

Output: `output/ratman_pose_pairs/`

For re-running failed pairs:
```bash
python3 tools/ratman_pose_pairs_rerun.py
```

### 4. Extract keypoints to .npy

Extracts raw DWPose keypoints from pose pairs as numpy arrays for training:

```bash
python3 tools/ratman_pose_npy_from_pairs.py
```

Output: `output/ratman_pose_npy/`

### 5. (Optional) Visualise keypoints

Preview a `.npy` keypoint file as a video:

```bash
python3 tools/npy_to_video.py
```

Edit `POSE_FILES` at the top of the script to point to your `.npy` files.

### 6. Generate pose videos from source video

To generate a DWPose skeleton video from an existing video (e.g. for inference):

```bash
python3 generate_training_poses.py
```

### 7. Fine-tune MusePose

Configs are in `MusePose/configs/`. Launch stage-1 training:

```bash
cd MusePose
accelerate launch train_stage_1_multiGPU.py --config configs/train_stage_1.yaml
```

For inference:
```bash
python3 MusePose/test_stage_2.py --config MusePose/configs/test_stage_2.yaml
```

## Key files

| File | Purpose |
|---|---|
| `tools/ratman_pose_images.py` | DWPose detection on character images |
| `tools/ratman_pose_pairs.py` | Assemble reference + pose training pairs |
| `tools/ratman_pose_npy_from_pairs.py` | Extract keypoints to `.npy` |
| `tools/npy_to_video.py` | Visualise `.npy` keypoints as video |
| `generate_training_poses.py` | Generate pose skeleton video from source video |
| `MusePose/train_stage_1_multiGPU.py` | Multi-GPU stage-1 fine-tuning |
