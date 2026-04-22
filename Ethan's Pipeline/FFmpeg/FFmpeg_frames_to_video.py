import os
import subprocess
import imageio_ffmpeg as ffmpeg

def frames_to_video(input_dir, output_path, fps=24):
    input_pattern = os.path.join(input_dir, "ai_generated_frame_%04d.png")
    ffmpeg_path = ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_path,
        "-y",
        "-framerate", str(fps),
        "-i", input_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_path
    ]

    try:
        subprocess.run(command, check=True)
        print(f"✅ Video saved to '{output_path}'")
    except subprocess.CalledProcessError as e:
        print("❌ Error while creating video:", e)

def get_video(input_dir):
    output_video = "videos/output.mp4"
    frames_to_video(input_dir, output_video, fps=24)

if __name__ == "__main__":
    get_video()


