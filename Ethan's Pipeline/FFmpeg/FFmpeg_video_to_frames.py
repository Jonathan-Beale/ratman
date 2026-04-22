import os
import subprocess
import json

def get_video_duration(video_path):
    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    data = json.loads(result.stdout)
    return float(data["format"]["duration"])

def extract_frames(video_path, num_frames):
    output_dir = "FFmpeg/FFmpeg Images"
    os.makedirs(output_dir, exist_ok=True)

    duration = get_video_duration(video_path)

    # calculate FPS needed to get exact number of frames
    fps = num_frames / duration

    output_pattern = os.path.join(output_dir, "frame_%04d.png")

    command = [
        "ffmpeg",
        "-i", video_path,
        #"-vf", f"fps={fps}",  # if you want all the frames just remove this line -> "-vf", f"fps={fps}"
        output_pattern
    ]

    try:
        subprocess.run(command, check=True)
        print(f"✅ Extracted {num_frames} frames into '{output_dir}'")
    except subprocess.CalledProcessError as e:
        print("❌ Error while extracting frames:", e)

def get_frames(video_name):
    extract_frames(video_name, num_frames=12)

if __name__ == "__main__":
    get_frames("FFmpeg/videos/input.mp4")