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
- DreamShaper 8 downloads automatically from HuggingFace on first run (~2GB, cached locally after that)

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
| `--lora_weights` | None | Path to trained LoRA weights directory |

### Quick test (3 frames)
```bash
python3 ratman_pipeline.py --input_video input.mp4 --reference_image superhero.png --output_video test.mp4 --max_frames 3
```

### With LoRA weights
```bash
python3 ratman_pipeline.py --input_video input.mp4 --reference_image superhero.png --lora_weights lora_weights/ --output_video output.mp4
```

## Output

- Individual frames saved to `output/frame_XXXX.png`
- Final assembled video at the path given by `--output_video`

## LoRA Fine-Tuning (optional — improves character consistency)

LoRA fine-tunes the model on your specific character so it generates them more reliably across frames.

### 1. Collect training images
Add 15-20 Batman images to `training_data/` — see `training_data/README.md` for guidance.

### 2. Run training (~3-6 hours on GPU)
```bash
python3 lora_train.py --instance_data_dir training_data/ --output_dir lora_weights/
```

Key options:
| Flag | Default | Description |
|---|---|---|
| `--num_train_steps` | `1000` | More steps = better quality (try 800-1500) |
| `--rank` | `8` | LoRA rank — higher = more expressive (8-16 recommended) |
| `--instance_prompt` | `batman, dark superhero suit...` | Caption applied to all training images |

### 3. Run pipeline with trained LoRA
```bash
python3 ratman_pipeline.py \
  --input_video input.mp4 \
  --reference_image superhero.png \
  --lora_weights lora_weights/ \
  --output_video output_lora.mp4
```
