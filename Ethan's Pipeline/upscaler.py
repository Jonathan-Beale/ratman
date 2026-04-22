import torch
import os
from PIL import Image
from diffusers import (
    AnimateDiffVideoToVideoControlNetPipeline, 
    ControlNetModel, 
    AutoencoderKL, 
    EulerDiscreteScheduler
)
from FFmpeg.FFmpeg_frames_to_video import get_video
get_video("upscaled_frames")
# =========================
# CONFIGURATION
# =========================
INPUT_DIR = "generated_frames"
POSE_DIR = "Openpose/results"
OUTPUT_DIR = "upscaled_frames"
MARVEL_DIFFUSERS_PATH = "marvels_dungeons_diffusers"
LORA_PATH = "SpidermanNoirLora.safetensors"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# PROMPTS
# =========================
COMIC_PROMPT = "spider-man noir, masterpiece, best quality, highly detailed, ultra-detailed, sharp focus, crisp lines, high contrast, 4k, comic book style, highly detailed suit texture"
NEGATIVE_PROMPT = "blurry, out of focus, soft, washed out, lowres, bad hands, smooth, easynegative, verybadimagenegative_v1.3, bad anatomy"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16
generator = torch.Generator(device=device).manual_seed(56461)

# =========================
# LOAD MODELS
# =========================
print("Loading Upscale Pipeline...")
controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=dtype)
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)

pipe = AnimateDiffVideoToVideoControlNetPipeline.from_pretrained(
    MARVEL_DIFFUSERS_PATH,
    controlnet=controlnet,
    vae=vae,
    torch_dtype=dtype
)

pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config,
    beta_schedule="linear",
    use_karras_sigmas=True,
    timestep_spacing="linspace",
    steps_offset=1
)

pipe.load_lora_weights(LORA_PATH, adapter_name="spiderman_noir")
pipe.set_adapters(["spiderman_noir"], adapter_weights=[1.0])
pipe.fuse_lora()

pipe.load_textual_inversion("easynegative.safetensors", token="easynegative")
pipe.load_textual_inversion("verybadimagenegative_v1.3.pt", token="verybadimagenegative_v1.3")
pipe.load_textual_inversion("negative_hand-neg.pt", token="negative_hand-neg")

pipe.to(device)
pipe.enable_model_cpu_offload() 
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

# =========================
# PROCESSING
# =========================
input_frames = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".png")])
pose_files = sorted([f for f in os.listdir(POSE_DIR) if f.startswith("pose_frame_")])

chunk_size = 16 
frame_counter = 1

print(f"Upscaling {len(input_frames)} frames...")

for i in range(0, (len(input_frames) // chunk_size) * chunk_size, chunk_size):
    chunk = input_frames[i : i + chunk_size]
    print(f"Processing chunk {i//chunk_size + 1}...")

    frames = [Image.open(os.path.join(INPUT_DIR, f)).convert("RGB").resize((1024, 1024)) for f in chunk]
    poses = [Image.open(os.path.join(POSE_DIR, f)).convert("RGB").resize((1024, 1024)) for f in pose_files[i : i + chunk_size]]

    # Video-to-Video Inference
    output = pipe(
        prompt=COMIC_PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        video=frames,                       
        conditioning_frames=poses,          
        strength=0.35,                      
        num_inference_steps=25,
        guidance_scale=7.5,
        controlnet_conditioning_scale=0.7,
        generator=generator,
        # FORCE DIMENSIONS HERE
        width=1024,
        height=1024
    ).frames[0]

    for frame in output:
        frame.save(os.path.join(OUTPUT_DIR, f"ai_generated_frame_{frame_counter:04d}.png"))
        frame_counter += 1

print(f"Done! Frames saved to {OUTPUT_DIR}")
get_video("upscaled_frames")