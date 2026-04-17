import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading YOLO models...")
seg_model = YOLO("yolov8n-seg.pt")
pose_model = YOLO("yolov8n-pose.pt")
seg_model.to(device)
pose_model.to(device)

print("Loading Stable Diffusion Pipeline...")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    controlnet=controlnet,
    torch_dtype=torch.float16,
)
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
pipe.set_ip_adapter_scale(0.6)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

inputFolder = "input"
outputFolder = "output"
os.makedirs(outputFolder, exist_ok=True)

superhero_img_path = "superhero.png"
if not os.path.exists(superhero_img_path):
    print(f"Warning: {superhero_img_path} not found.")
    superhero_img = Image.new("RGB", (512, 512), "red")
else:
    superhero_img = Image.open(superhero_img_path).convert("RGB")
superhero_img = superhero_img.resize((512, 512))

def draw_openpose_skeleton(keypoints, width, height):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if keypoints is None or len(keypoints) == 0:
        return canvas
    
    kpts = keypoints[0].cpu().numpy()
    stickwidth = 4
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
              
    pairs = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
    
    for i, (p1, p2) in enumerate(pairs):
        if kpts[p1][2] > 0.3 and kpts[p2][2] > 0.3:
            pt1 = (int(kpts[p1][0]), int(kpts[p1][1]))
            pt2 = (int(kpts[p2][0]), int(kpts[p2][1]))
            cv2.line(canvas, pt1, pt2, colors[i % len(colors)], stickwidth)
            cv2.circle(canvas, pt1, 4, colors[i % len(colors)], thickness=-1)
            cv2.circle(canvas, pt2, 4, colors[i % len(colors)], thickness=-1)
            
    return canvas

for filename in os.listdir(inputFolder):
    if filename.lower().endswith((".mp4", ".mov")):
        input_path = os.path.join(inputFolder, filename)
        name = os.path.splitext(filename)[0]
        output_path = os.path.join(outputFolder, f"{name}_superhero_overlay.mp4")
        
        print(f"Processing {filename}...")
        
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
        
        kernel = np.ones((15, 15), np.uint8)
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            if os.getenv('TEST_RUN') and frame_idx > 2:
                break
            
            print(f"Processing frame {frame_idx}...")
            seg_results = seg_model(frame, classes=[0], verbose=False, device=device)
            mask_8u = np.zeros((height, width), dtype=np.uint8)
            if seg_results[0].masks is not None:
                mask_tensor = seg_results[0].masks.data[0]
                mask_numpy = mask_tensor.cpu().numpy()
                mask_resized = cv2.resize(mask_numpy, (width, height))
                mask_8u = (mask_resized * 255).astype(np.uint8)
                mask_8u = cv2.dilate(mask_8u, kernel, iterations=2)
            
            pose_results = pose_model(frame, classes=[0], verbose=False, device=device)
            keypoints = pose_results[0].keypoints.data if pose_results[0].keypoints is not None else None
            
            pose_img_cv = draw_openpose_skeleton(keypoints, width, height)
            pose_image = Image.fromarray(cv2.cvtColor(pose_img_cv, cv2.COLOR_BGR2RGB))
            
            sd_size = (512, 512)
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize(sd_size)
            mask_pil = Image.fromarray(mask_8u).resize(sd_size)
            pose_image_resized = pose_image.resize(sd_size)
            
            gen_img = pipe(
                prompt="a superhero in a dynamic pose, highly detailed, realistic, best quality",
                negative_prompt="low quality, blurry, mutated, ugly, bad anatomy",
                image=frame_pil,
                mask_image=mask_pil,
                control_image=pose_image_resized,
                ip_adapter_image=superhero_img,
                num_inference_steps=20,
                guidance_scale=7.5
            ).images[0]
            
            gen_img_resized = gen_img.resize((width, height))
            gen_img_cv = cv2.cvtColor(np.array(gen_img_resized), cv2.COLOR_RGB2BGR)
            
            mask_3ch = cv2.cvtColor(mask_8u, cv2.COLOR_GRAY2BGR) / 255.0
            composite_frame = (gen_img_cv * mask_3ch + frame * (1 - mask_3ch)).astype(np.uint8)
            
            out.write(composite_frame)
            
        cap.release()
        out.release()
        print(f"Saved superhero overlay video to: {output_path}")

print("Superhero video generation complete!")
