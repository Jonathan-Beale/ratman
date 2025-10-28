import os
import subprocess

def extract_frames(video_path):
    output_dir = "FFmpeg Images"

    # Build ffmpeg command
    # the frames will be saved as frame_0001.png, frame_0002.png, etc.
    output_pattern = os.path.join(output_dir, "frame_%04d.png")
    command = [
        "ffmpeg",
        "-i", video_path,   # input video file
        output_pattern      # output file pattern
    ]

    # Run the command
    try:
        subprocess.run(command, check=True)
        print(f"✅ Frames saved in '{output_dir}'")
    except subprocess.CalledProcessError as e:
        print("❌ Error while extracting frames:", e)

# main function to test the frame extraction
if __name__ == "__main__":
    video_file = "videos/Output_20251027_123030.mp4" # change this to your video file
    extract_frames(video_file)