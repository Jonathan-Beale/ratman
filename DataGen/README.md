# DataGen

Synthetic dataset generation pipeline using Blender. Renders 3D character animations from multiple camera angles and exports ground-truth DWPose skeleton data for MusePose fine-tuning.

## Overview

```
Assets/
  Animations/   FBX/BVH animation clips
  Models/       FBX/GLB/blend character rigs
  Scenes/       .blend scene files (cameras, lighting, environment)
Renders/        Output — organised by model/animation/scene/camera
```

The pipeline loads each (model, animation, scene) combination into Blender, retargets the animation onto the model, renders all scene cameras, and exports:

- `{cam}.mp4` — raw RGB render
- `{cam}_overlay.mp4` — skeleton overlay on the render
- `{cam}_dwpose.mp4` — custom red/green DWPose skeleton (for visualisation)
- `{cam}_dwpose_rainbow.mp4` — standard MusePose rainbow skeleton (for inference/training)
- `reference.png` — single rest-pose front-facing shot of the model (single-actor only)
- `skeleton_coco.npz` — ground-truth COCO-17 + hands keypoints per frame
- `skeleton_h36m.npz` — ground-truth H36M-17 keypoints per frame

## Usage

### 1. Add assets

Place files in the appropriate `Assets/` subdirectory. Supported formats:

| Type | Formats |
|---|---|
| Scenes | `.blend` |
| Models | `.fbx`, `.glb`, `.gltf`, `.blend` |
| Animations | `.fbx`, `.bvh` |

### 2. Configure

Edit the top of `dataset_pipeline.py`:

```python
NUM_ACTORS = 1            # 1 = single-actor, 2+ = multi-actor combos
DELETE_FRAMES_AFTER_COMPILE = True  # delete PNGs after compiling .mp4s
```

### 3. Run

```bash
python3 dataset_pipeline.py
```

Blender and ffmpeg are expected on Windows (`C:\Program Files\...`). The script runs from WSL2 and uses `wslpath` to translate Linux paths for Windows processes.

### 4. Build MusePose dataset

Once renders are complete, run `build_musepose_dataset.py` to organise outputs into the layout expected by MusePose fine-tuning:

```bash
python3 build_musepose_dataset.py
```

This produces:
```
musepose_dataset/
  dataset/                   raw RGB mp4s
  dataset_dwpose/            DWPose skeleton mp4s
  dataset_dwpose_keypoints/  per-video .npy keypoint files
  meta/dataset.json
```

## Key files

| File | Purpose |
|---|---|
| `dataset_pipeline.py` | Orchestrates Blender jobs and ffmpeg compilation |
| `blender_render_job.py` | Blender-side script — retargeting, rendering, skeleton export |
| `build_musepose_dataset.py` | Converts renders into MusePose training layout |
