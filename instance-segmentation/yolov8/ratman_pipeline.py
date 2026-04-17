import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, DPMSolverMultistepScheduler
from ultralytics import YOLO

_HERE = os.path.dirname(os.path.abspath(__file__))

YOLO_MODEL_PATH = os.path.join(_HERE, "yolo12l-person-seg.pt")
SD_BASE_MODEL_PATH = "Lykon/dreamshaper-8"
SD_LOCAL_ONLY = False
OUTPUT_DIR = os.path.join(_HERE, "output")

NUM_INFERENCE_STEPS = 25       # DPM++ 25 steps — high quality overnight run
GUIDANCE_SCALE = 10.0
CONTROLNET_SCALE = 0.85        # stronger body structure adherence
IP_ADAPTER_SCALE = 0.85        # stronger Batman appearance reference
SEED = 1024
CONTROLNET_SIZE = (384, 384)
OUTPUT_FPS = 24
ANIMATE_ON_DOUBLES = True      # sample at ~12fps, hold each frame for 2 output frames
MASK_DILATION_PX = 40          # pixels to expand person mask — gives cape room to show

# Canny thresholds for edge detection on the masked person
CANNY_LOW = 50
CANNY_HIGH = 150


def get_device_and_dtype():
    if torch.cuda.is_available():
        # Turing GPUs (GTX 16xx/RTX 20xx, compute cap 7.5) have no flash attention.
        # Without flash attention, float16 math SDP produces NaN → black frames.
        # float32 is actually FASTER on these GPUs (~1s/step vs ~3.5s/step)
        # and fits in 6 GB with SD v1.5 + ControlNet + IP-Adapter.
        return "cuda", torch.float32
    if torch.backends.mps.is_available():
        return "mps", torch.float32   # float16 has bugs on MPS
    return "cpu", torch.float32


def load_models(lora_weights_path=None):
    device, dtype = get_device_and_dtype()
    print(f"Using device: {device} ({dtype})")

    print("Loading YOLO model...")
    yolo_model = YOLO(YOLO_MODEL_PATH)

    print("Loading ControlNet (canny)...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=dtype,
    ).to(device)

    print("Loading Stable Diffusion pipeline...")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD_BASE_MODEL_PATH,
        controlnet=controlnet,
        torch_dtype=dtype,
        local_files_only=SD_LOCAL_ONLY,
    ).to(device)

    # Swap to DPM++ solver — same quality as DDIM @ 20 steps but done in 8
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        final_sigmas_type="sigma_min",
    )

    pipe.safety_checker = None
    # Attention slicing reduces peak VRAM — helps fit ControlNet+IP-Adapter in 6 GB
    pipe.enable_attention_slicing(1)

    print("Loading IP-Adapter weights...")
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)

    if lora_weights_path:
        print(f"Loading LoRA weights from {lora_weights_path}...")
        pipe.load_lora_weights(lora_weights_path, weight_name="pytorch_lora_weights.safetensors")
        pipe.fuse_lora(lora_scale=0.8)
        print("LoRA fused into pipeline.")

    return yolo_model, pipe


def open_video(input_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, width, height, fps, total


def make_canny_control(results, frame_bgr):
    """Build a canny edge map for ControlNet.

    Runs Canny on the FULL frame (real contrast, real edges) then masks the
    result to the person region.  Also draws the person silhouette outline
    explicitly so ControlNet always sees the body shape even when clothing
    blends into the background (e.g. dark hoodie on dark bg)."""
    h, w = frame_bgr.shape[:2]

    person_mask = np.zeros((h, w), dtype=np.uint8)
    if results[0].masks is not None:
        for mask_tensor in results[0].masks.data:
            mask = mask_tensor.cpu().numpy()
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            person_mask[mask_resized > 0.5] = 255

    # Canny on the full frame — preserves real contrast between person and background
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # Boost contrast with CLAHE before Canny so dark clothing edges still show
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    edges = cv2.Canny(gray_eq, CANNY_LOW, CANNY_HIGH)

    # Keep only edges that fall inside the person mask
    edges_masked = cv2.bitwise_and(edges, edges, mask=person_mask)

    # Draw the silhouette outline explicitly (1-px dilation − erosion = boundary)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    outline = cv2.morphologyEx(person_mask, cv2.MORPH_GRADIENT, kernel)
    # Merge: interior texture edges + hard silhouette border
    edges_final = cv2.bitwise_or(edges_masked, outline)

    edges_rgb = cv2.cvtColor(edges_final, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb), person_mask


def composite_onto_background(generated_pil, person_mask, original_frame_bgr, target_w, target_h):
    """Cut the generated character out using the YOLO mask and paste it
    onto the original video frame background."""
    generated_bgr = cv2.cvtColor(np.array(generated_pil), cv2.COLOR_RGB2BGR)
    generated_bgr = cv2.resize(generated_bgr, (target_w, target_h))
    mask_resized = cv2.resize(person_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    # Dilate mask so cape/costume elements outside the bare body silhouette show through
    if MASK_DILATION_PX > 0:
        k = MASK_DILATION_PX * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_resized = cv2.dilate(mask_resized, kernel)
    background = cv2.resize(original_frame_bgr, (target_w, target_h))

    # Blend: generated character where mask, original video everywhere else
    mask_3ch = cv2.merge([mask_resized, mask_resized, mask_resized]).astype(np.float32) / 255.0
    composite = (generated_bgr.astype(np.float32) * mask_3ch +
                 background.astype(np.float32) * (1.0 - mask_3ch))
    return composite.astype(np.uint8)


def run_pipeline(cap, yolo_model, sd_pipe, ref_image, total_frames, target_w, target_h, input_fps, max_frames=0):
    device, _ = get_device_and_dtype()
    generator = None
    output_frames = []
    frame_idx = 0

    # Sample input at 12fps to animate on doubles at 24fps output
    sample_every = max(1, round(input_fps / 12)) if ANIMATE_ON_DOUBLES else 1
    frames_to_process = (total_frames // sample_every) if total_frames else "?"
    print(f"Input FPS: {input_fps} — sampling every {sample_every} frame(s) → ~12 unique frames/sec → 24fps output on doubles")
    print(f"Frames to generate: {frames_to_process}")

    prev_generated = None
    anchor_frame = None     # first generated frame used as style anchor
    BLEND_ALPHA = 0.15      # % of previous frame to mix in (reduces flicker)

    raw_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        raw_idx += 1
        if (raw_idx - 1) % sample_every != 0:
            continue

        frame_idx += 1
        if max_frames > 0 and frame_idx > max_frames:
            break
        print(f"Processing frame {frame_idx}/{frames_to_process}...")

        # Reset seed every frame so each starts from identical random noise.
        # Without this the generator state drifts, making each Batman different.
        generator = torch.Generator(device=device).manual_seed(SEED)

        results = yolo_model(frame, verbose=False)

        # Canny edge map of the person region → ControlNet structure input
        canny_pil, person_mask = make_canny_control(results, frame)
        canny_pil = canny_pil.resize(CONTROLNET_SIZE, Image.NEAREST)

        # IP-Adapter input: Batman reference for frame 1, then blend 20% of the
        # first generated frame into the reference to anchor the style across frames.
        if anchor_frame is None:
            ip_input = ref_image
        else:
            ref_arr = np.array(ref_image, dtype=np.float32)
            anchor_arr = np.array(anchor_frame.resize(CONTROLNET_SIZE), dtype=np.float32)
            ip_input = Image.fromarray((0.8 * ref_arr + 0.2 * anchor_arr).astype(np.uint8))

        generated = sd_pipe(
            prompt="BatmanCore, batman costume, dark superhero suit, black cape, yellow utility belt, masked hero, comic book style, masterpiece, best quality, highly detailed",
            negative_prompt="bad quality, messy, glitch, blurry, deformed, extra limbs, watermark, cartoon, anime",
            image=canny_pil,
            ip_adapter_image=ip_input,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            controlnet_conditioning_scale=CONTROLNET_SCALE,
            generator=generator,
        ).images[0]

        # Store first generated frame as the style anchor
        if anchor_frame is None:
            anchor_frame = generated

        # Blend a small amount of the previous generated frame to reduce flicker
        if prev_generated is not None:
            gen_arr = np.array(generated, dtype=np.float32)
            prev_arr = np.array(prev_generated, dtype=np.float32)
            blended = (1.0 - BLEND_ALPHA) * gen_arr + BLEND_ALPHA * prev_arr
            generated = Image.fromarray(blended.astype(np.uint8))

        prev_generated = generated

        # Resize generated frame to target resolution and convert to BGR for VideoWriter
        out_pil = generated.resize((target_w, target_h), Image.LANCZOS)
        out_bgr = cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)

        hold = 2 if ANIMATE_ON_DOUBLES else 1
        for _ in range(hold):
            output_frames.append(out_bgr)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_pil.save(os.path.join(OUTPUT_DIR, f"frame_{frame_idx:04d}.png"))

    cap.release()
    return output_frames


def assemble_video(output_frames, fps, output_path, target_width, target_height):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

    for bgr_frame in output_frames:
        writer.write(bgr_frame)

    writer.release()
    print(f"Output video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Ratman end-to-end video generation pipeline")
    parser.add_argument("--input_video", required=True, help="Path to input video")
    parser.add_argument("--reference_image", required=True, help="Path to reference character image")
    parser.add_argument("--output_video", default="output_final.mp4", help="Path for output video")
    parser.add_argument("--max_frames", type=int, default=0, help="Stop after N frames (0 = all)")
    parser.add_argument("--lora_weights", default=None, help="Path to trained LoRA weights directory (optional)")
    args = parser.parse_args()

    if not os.path.isfile(args.input_video):
        raise FileNotFoundError(f"Input video not found: {args.input_video}")
    if not os.path.isfile(args.reference_image):
        raise FileNotFoundError(f"Reference image not found: {args.reference_image}")

    yolo_model, sd_pipe = load_models(lora_weights_path=args.lora_weights)

    cap, width, height, fps, total_frames = open_video(args.input_video)
    ref_image = Image.open(args.reference_image).convert("RGB").resize(CONTROLNET_SIZE)

    output_frames = run_pipeline(cap, yolo_model, sd_pipe, ref_image, total_frames, width, height, fps, args.max_frames)

    assemble_video(output_frames, OUTPUT_FPS, args.output_video, width, height)


if __name__ == "__main__":
    main()
