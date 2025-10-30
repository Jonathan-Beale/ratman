# Advanced Stable Diffusion Implementations

This directory contains three advanced implementations for AI-powered image and video generation:

1. **ControlNet + Stable Diffusion** - Pose-guided image generation
2. **FLUX** - State-of-the-art high-quality generation
3. **Pose-to-Video Pipeline** - Complete temporal generation system

---

## 1. ControlNet Pose-Guided Generation

**File:** `controlnet_pose_generation.py`

### What it does
Generates images that follow specific human poses using OpenPose skeleton data. Perfect for:
- Creating character animations
- Generating training data
- Consistent character poses across frames
- Integrating with pose estimation systems

### Features
- Accepts pose keypoints directly from pose estimation
- Compatible with COCO keypoint format
- Generates images matching specific poses
- Can use existing pose estimation code in `/pose-estimation`

### Usage

#### Basic Example
```python
from controlnet_pose_generation import PoseGuidedGenerator
import numpy as np

# Initialize generator
generator = PoseGuidedGenerator()

# Define keypoints (COCO format: 17 keypoints)
keypoints = np.array([
    [x1, y1, confidence],  # nose
    [x2, y2, confidence],  # left eye
    # ... 15 more keypoints
])

# Generate image
output_path = generator.generate_from_keypoints(
    keypoints=keypoints,
    prompt="a superhero in dynamic pose, detailed costume, 4k",
    num_inference_steps=30,
    guidance_scale=7.5
)
```

#### Integration with Existing Pose Estimation
```python
# Example: Using with MediaPipe or HRNet output
import sys
sys.path.append('../pose-estimation/MediaPipe')
from mediapipe_inference import detect_pose  # Your existing code

# Get pose from image
image_path = "person.jpg"
keypoints = detect_pose(image_path)  # Your pose detection

# Generate styled image with same pose
generator = PoseGuidedGenerator()
output = generator.generate_from_keypoints(
    keypoints=keypoints,
    prompt="cyberpunk character, neon lighting, futuristic"
)
```

### Running the Demo
```bash
python3 controlnet_pose_generation.py
```

### Requirements
- GPU: 6GB+ VRAM (8GB recommended)
- Model size: ~5GB download
- Inference time: ~5-10 seconds per image

---

## 2. FLUX High-Quality Generation

**File:** `flux_generation.py`

### What it does
Uses FLUX from Black Forest Labs (creators of original Stable Diffusion) for state-of-the-art image quality. FLUX produces:
- Exceptional detail and realism
- Better prompt understanding
- Higher resolution (1024x1024 native)
- Superior composition

### Two Variants

#### FLUX.1-schnell (Fast)
```python
from flux_generation import FluxGenerator

generator = FluxGenerator(model_id="black-forest-labs/FLUX.1-schnell")

# Fast generation (4 steps)
output_paths = generator.generate(
    prompt="Batman on rooftop, Gotham cityscape, cinematic, 8k",
    height=1024,
    width=1024,
    num_inference_steps=4,  # Optimal for schnell
    guidance_scale=0.0,      # Schnell doesn't use CFG
    num_images=1
)
```

#### FLUX.1-dev (High Quality)
```python
generator = FluxGenerator(model_id="black-forest-labs/FLUX.1-dev")

# Higher quality (more steps)
output_paths = generator.generate(
    prompt="ultra detailed portrait, dramatic lighting, photorealistic",
    height=1024,
    width=1024,
    num_inference_steps=50,
    guidance_scale=3.5,
    num_images=1
)
```

### Running the Demo
```bash
python3 flux_generation.py
```

### Requirements
- GPU: 16GB+ VRAM (24GB recommended)
- Model size: ~24GB download
- Inference time:
  - schnell: ~10-15 seconds per image
  - dev: ~30-60 seconds per image

### Notes
- FLUX requires accepting license on HuggingFace
- May require HuggingFace token for access
- Significantly higher quality than SD 1.5
- Not compatible with ControlNet yet (as of 2025)

---

## 3. Pose-to-Video Pipeline

**File:** `pose_to_video_pipeline.py`

### What it does
Complete pipeline combining:
1. Pose estimation data
2. ControlNet for pose-guided generation
3. Temporal consistency for video

Perfect for:
- Generating character animations from pose sequences
- Creating consistent video from motion capture
- Animating characters with specific movements

### Features
- Accepts sequences of pose keypoints
- Generates temporally consistent frames
- Exports to video format
- Optional AnimateDiff integration for smoother motion

### Usage

#### Generate Video from Pose Sequence
```python
from pose_to_video_pipeline import PoseToVideoGenerator

# Initialize
generator = PoseToVideoGenerator(use_controlnet=True)

# Define pose sequence (list of keypoint arrays)
pose_sequence = [
    pose_frame1,  # keypoints at t=0
    pose_frame2,  # keypoints at t=1
    pose_frame3,  # keypoints at t=2
    # ... more frames
]

# Generate video
video_path = generator.generate_video_from_pose_sequence(
    pose_sequence=pose_sequence,
    prompt="athletic person, gym environment, professional lighting"
)
```

#### With AnimateDiff (Text-to-Video)
```python
# Generate video directly from text
video_path = generator.generate_video_with_animatediff(
    prompt="a person doing jumping jacks, exercise video, bright lighting",
    num_frames=16,
    num_inference_steps=25
)
```

### Running the Demo
```bash
python3 pose_to_video_pipeline.py
```

### Requirements
- GPU: 8GB+ VRAM (12GB recommended)
- Model size: ~10GB download (with AnimateDiff)
- Processing time: ~10-15 seconds per frame

---

## Comparison Table

| Feature | ControlNet | FLUX | Pose-to-Video |
|---------|-----------|------|---------------|
| **Quality** | Good | Excellent | Good |
| **Speed** | Fast | Medium | Slow |
| **VRAM** | 6GB | 16GB+ | 8GB+ |
| **Pose Control** | ✅ Yes | ❌ No | ✅ Yes |
| **Video Output** | ❌ No | ❌ No | ✅ Yes |
| **Resolution** | 512x512 | 1024x1024 | 512x512 |
| **Use Case** | Pose-guided images | Best quality images | Animation/video |

---

## Installation

### Install all dependencies
```bash
cd stable-diffusion
pip install -r requirements.txt
```

### Additional dependencies (if needed)
```bash
# For AnimateDiff
pip install imageio imageio-ffmpeg

# For FLUX (if using)
pip install sentencepiece protobuf
```

---

## Recommended Workflows

### Workflow 1: Pose-Guided Character Generation
Best for creating consistent character images with specific poses.

```
Your Pose Estimation → ControlNet → Styled Images
(HRNet/MediaPipe)
```

**Use:** `controlnet_pose_generation.py`

### Workflow 2: Maximum Quality Images
Best for final renders, marketing materials, high-quality outputs.

```
Text Prompt → FLUX → Ultra High-Quality Image
```

**Use:** `flux_generation.py`

### Workflow 3: Character Animation
Best for creating animated sequences with motion.

```
Pose Sequence → ControlNet → Frame Sequence → Video
(from motion data)
```

**Use:** `pose_to_video_pipeline.py`

### Workflow 4: Combined Pipeline
Best for production-quality character animations.

```
Pose Estimation → ControlNet → Frames → Post-process → Final Video
                                        ↓
                              (Optional: Upscale with FLUX)
```

---

## Integration Examples

### Example 1: Using Existing Pose Estimation
```python
# Import your existing pose code
import sys
sys.path.append('../pose-estimation/MediaPipe')
from mediapipe_inference import PoseDetector

# Import ControlNet generator
from controlnet_pose_generation import PoseGuidedGenerator

# Setup
pose_detector = PoseDetector()
image_generator = PoseGuidedGenerator()

# Process video frames
video_frames = load_video("input.mp4")
for frame in video_frames:
    # Detect pose
    keypoints = pose_detector.detect(frame)

    # Generate styled version
    styled_frame = image_generator.generate_from_keypoints(
        keypoints=keypoints,
        prompt="animated character, cartoon style"
    )

    # Save or process further
    save_frame(styled_frame)
```

### Example 2: Batch Processing
```python
from flux_generation import FluxGenerator

generator = FluxGenerator()

# Generate multiple variations
prompts = [
    "Batman in rain, dramatic lighting",
    "Batman on rooftop, sunset",
    "Batman in Batcave, tech lighting"
]

for prompt in prompts:
    generator.generate(prompt, num_images=3)
```

---

## Troubleshooting

### Out of Memory Errors
```python
# Enable memory optimizations
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()
```

### Slow Generation
- Use fewer inference steps (minimum: 20-25)
- Reduce image resolution
- Use FLUX.1-schnell instead of dev
- Enable attention slicing

### FLUX Access Issues
1. Visit HuggingFace model page
2. Accept model license
3. Generate access token
4. Login: `huggingface-cli login`

---

## Future Enhancements

Potential additions:
- FLUX ControlNet (when officially released)
- Real-time generation optimization
- Multi-person pose support
- Style transfer integration
- Upscaling pipeline
- Background replacement

---

## Performance Tips

1. **First Run:** Models download automatically (one-time, ~5-30 minutes)
2. **Subsequent Runs:** Much faster using cached models
3. **Batch Processing:** Generate multiple images at once for efficiency
4. **Mixed Precision:** Use `torch.float16` on GPU for speed
5. **VRAM Management:** Close other GPU-intensive applications

---

## License & Credits

- **Stable Diffusion:** Licensed under CreativeML OpenRAIL-M
- **ControlNet:** Apache 2.0
- **FLUX:** Check HuggingFace for specific terms
- **AnimateDiff:** MIT License

Always respect model licenses and usage terms.
