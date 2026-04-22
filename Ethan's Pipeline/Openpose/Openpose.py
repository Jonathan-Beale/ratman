from controlnet_aux import OpenposeDetector
from PIL import Image
import os

def run_openpose():
    detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")  

    input_folder = "FFmpeg/FFmpeg Images"
    output_folder = "Openpose/results/"
    os.makedirs(output_folder, exist_ok=True)

    for i in range(len(os.listdir(input_folder))):
        input_image = Image.open(
            f"{input_folder}/frame_{i+1:04d}.png"
        ).convert("RGB")
        input_image = input_image.resize((1024, 1024))

        pose_image = detector(input_image)

        pose_image.save(f"{output_folder}/pose_frame_{i+1:04d}.png")
        print(f"Saved pose_frame_{i+1:04d}.png")

        print("✅ Openpose skeletons saved")


if __name__ == "__main__":
    run_openpose()

    