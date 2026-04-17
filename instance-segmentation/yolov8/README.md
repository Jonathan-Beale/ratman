# Ratman Pipeline — YOLOv12 + Stable Diffusion Superhero Overlay

Takes an input video and a reference character image and outputs a video where the person has been replaced with the reference character (e.g. Batman), preserving their pose and motion.

## How it works

1. **YOLOv12-Seg** — segments the person in each frame
2. **Canny edge detection** (CLAHE-enhanced) — extracts body structure for ControlNet
3. **DreamShaper 8** (SD 1.5 fine-tune) via ControlNet + IP-Adapter — generates the character in the correct pose
4. **Temporal consistency** — fixed seed + frame blending keeps the character stable across frames

## Setup

```bash
./setup.sh
source venv/bin/activate
```

You will also need:
- `yolo12l-person-seg.pt` — place in this directory (excluded from repo, ~58MB)
- On Mac: a local SD 1.5 model at `~/Projects/cleanRoom/magic_code/pretrained_models/stable-diffusion-v1-5`
- On Windows: DreamShaper 8 downloads automatically from HuggingFace on first run

## Usage (CLI)

```bash
python3 ratman_pipeline.py \
  --input_video path/to/video.mp4 \
  --reference_image path/to/character.png \
  --output_video output.mp4
```

### Optional flags
| Flag | Default | Description |
|---|---|---|
| `--output_video` | `output_final.mp4` | Output file path |
| `--max_frames` | `0` (all) | Stop after N frames — useful for quick tests |

### Quick test (3 frames)
```bash
python3 ratman_pipeline.py --input_video input.mp4 --reference_image superhero.png --output_video test.mp4 --max_frames 3
```

## Output

- Individual frames saved to `output/frame_XXXX.png`
- Final assembled video at the path given by `--output_video`
