import torch
from diffusers import (
    AnimateDiffControlNetPipeline, 
    ControlNetModel, 
    MotionAdapter, 
    AutoencoderKL, 
    EulerDiscreteScheduler
)
from PIL import Image
import os
import numpy as np
from FFmpeg.FFmpeg_video_to_frames import get_frames
from FFmpeg.FFmpeg_frames_to_video import get_video
from Openpose.Openpose import run_openpose

Height = 768
Width = 768


# =========================
# CLEAN PREVIOUS FRAMES
# =========================
for f in os.listdir("FFmpeg/FFmpeg Images"):
    os.remove(os.path.join("FFmpeg/FFmpeg Images", f))
for f in os.listdir("Openpose/results"):
    os.remove(os.path.join("Openpose/results", f))
for f in os.listdir("generated_frames"):
    os.remove(os.path.join("generated_frames", f))
for f in os.listdir("upscaled_frames"):
    os.remove(os.path.join("upscaled_frames", f))
    
# =========================
# EXTRACT FRAMES + POSES
# =========================
get_frames("FFmpeg/videos/input.mp4")
run_openpose()


print(torch.cuda.get_device_properties(0).total_memory / 1024**3)
print(torch.version.cuda)

# =========================
# PROMPTS
# =========================
COMIC_PROMPT = """ratman, highly detailed, brown hooded cloak, yellow rat logo on chest, 
face mask, glowing eyes, tactical belt, boots, comic book style, 
masterpiece, best quality, 4k, black background"""

NEGATIVE_PROMPT = """easynegative, verybadimagenegative_v1.3, negative_hand-neg, background, detailed background, outdoor, indoor, scenery, buildings, city, rain, dynamic background, colorful background, blurry, low quality, bad anatomy, extra limbs, worst quality"""

# =========================
# SETTINGS
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
print(f"Using device: {device}")
os.makedirs("generated_frames", exist_ok=True)

# =========================
# LOAD MOTION ADAPTER
# =========================
adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=dtype)

# =========================
# LOAD CONTROLNET
# =========================
controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=dtype)

# =========================
# LOAD VAE
# =========================
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)

# =========================
# CONVERT Marvel Diffuser MIX (ONE-TIME ONLY)
# =========================
MARVEL_SAFETENSOR_PATH = "M4RV3LSDUNGEONSNEWV40COMICS_mD40.safetensors"
MARVEL_DIFFUSERS_PATH = "marvels_dungeons_diffusers"
if not os.path.exists(MARVEL_DIFFUSERS_PATH):
    print("Converting Marvel Diffuser to diffusers format (one-time)...")
    from diffusers import StableDiffusionPipeline
    pipe_temp = StableDiffusionPipeline.from_single_file(MARVEL_SAFETENSOR_PATH, torch_dtype=dtype)
    pipe_temp.save_pretrained(MARVEL_DIFFUSERS_PATH)
    del pipe_temp
    torch.cuda.empty_cache()
    print("Conversion complete!")
else:
    print("Marvel diffuser already converted, skipping...")

# =========================
# LOAD ANIMATEDIFF PIPELINE
# =========================
pipe = AnimateDiffControlNetPipeline.from_pretrained(
    MARVEL_DIFFUSERS_PATH, 
    controlnet=controlnet, 
    motion_adapter=adapter, 
    vae=vae, 
    torch_dtype=dtype
)

pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config,
    beta_schedule="linear",        # Essential for motion stability in v1.5
    use_karras_sigmas=True,       # Helps prevent color washing/graying
    timestep_spacing="linspace",  # Ensures smooth frame-to-frame motion
    steps_offset=1
)

# =========================
# LOAD MULTIPLE LORAS
# =========================
pipe.load_lora_weights("Ratman_v1.safetensors", adapter_name="ratman")
pipe.set_adapters(["ratman"], adapter_weights=[1.0])
pipe.fuse_lora()

# =========================
# OPTIMIZATIONS (16GB VRAM)
# =========================
pipe.to(device)
pipe.load_textual_inversion("easynegative.safetensors", token="easynegative")
pipe.load_textual_inversion("verybadimagenegative_v1.3.pt", token="verybadimagenegative_v1.3")
pipe.load_textual_inversion("negative_hand-neg.pt", token="negative_hand-neg")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()


# =========================
# LOAD POSE FILES
# =========================
pose_dir = "Openpose/results/"
pose_files = sorted([f for f in os.listdir(pose_dir) if f.startswith("pose_frame_")])
total_poses = len(pose_files)
print(f"Found {total_poses} pose images")

# =========================
# PROCESS IN CHUNKS OF 16
# =========================
how_many_frames_to_generate = (len(pose_files) // 16) * 16
chunk_size = 16
chunks = [pose_files[i:i+chunk_size] for i in range(0, how_many_frames_to_generate, chunk_size)]
generator = torch.Generator(device=device).manual_seed(56461)
frame_counter = 1
for chunk_idx, chunk in enumerate(chunks):
    print(f"Generating chunk {chunk_idx+1}/{len(chunks)} ({len(chunk)} frames)...")
    conditioning_frames = [Image.open(os.path.join(pose_dir, f)).convert("RGB").resize((Width, Height)) for f in chunk]
    output = pipe(
        prompt=COMIC_PROMPT, 
        negative_prompt=NEGATIVE_PROMPT, 
        conditioning_frames=conditioning_frames, 
        num_frames=len(chunk), 
        width=Width, 
        height=Height, 
        num_inference_steps=25, 
        guidance_scale=7.5,             
        controlnet_conditioning_scale=1.3,
        generator=generator
    )
    for frame in output.frames[0]:
        output_path = f"generated_frames/ai_generated_frame_{frame_counter:04d}.png"
        frame.save(output_path)
        print(f"Saved frame {frame_counter}")
        frame_counter += 1
print(f"\n✅ Done! Generated {frame_counter-1} frames with Marvel Diffuser + AnimateDiff + ControlNet.")
get_video("generated_frames")