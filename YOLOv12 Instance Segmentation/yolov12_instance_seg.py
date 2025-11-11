from ultralytics import YOLO    # YOLOv12 framework - main interface for YOLO models
import cv2                      # Image I/O and display - reading and displaying images
import matplotlib.pyplot as plt # Plotting masks and results
import numpy as np              # For array manipulations (masks are often numpy arrays)
import os

# there needs to be a directory called "input_videos" that contains all the videos 
# you want the model to be run on.
# there should also be a directory called "output_videos" that is where all the
# output videos will go.

# YOLOv12 model I found on hugging face
model = YOLO("yolo12l-person-seg.pt")

#input and output directories
input_folder = "input_videos"
output_folder = "output_videos"
os.makedirs(output_folder, exist_ok=True)

# performs instance seg. on all video files in input_videos directory
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".mp4", ".avi", ".mov")): # accepts .mp4, .avi, and .mov files
        input_path = os.path.join(input_folder, filename)

        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_folder, f"{base_name}_boxes_masks.mp4")

        # --- Open video and prepare writer ---
        cap = cv2.VideoCapture(input_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"), # output file is a .mp4
            fps,
            (frame_width, frame_height)
        )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False) # remove 'verbose=False' to see details for each frame
            annotated_frame = results[0].plot()  # masks + boxes
            out.write(annotated_frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Output saved to {output_path}")