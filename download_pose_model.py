"""
Download MediaPipe pose estimation model
"""

import urllib.request
import os

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_NAME = "pose_landmarker_lite.task"

def download_model():
    if os.path.exists(MODEL_NAME):
        print(f"✓ Model '{MODEL_NAME}' already exists.")
        return
    
    print(f"Downloading {MODEL_NAME}...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_NAME)
        print(f"✓ Successfully downloaded '{MODEL_NAME}'")
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        print(f"Manual download: {MODEL_URL}")

if __name__ == "__main__":
    download_model()
