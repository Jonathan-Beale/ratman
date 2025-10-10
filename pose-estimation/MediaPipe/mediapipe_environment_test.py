# This is just a test script to ensure that the MediaPipe environment is set up correctly.
# It checks the versions of OpenCV, NumPy, and MediaPipe, and runs a simple inference

import mediapipe as mp
import cv2
import numpy as np

print("Testing clean MediaPipe environment...\n")
print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"MediaPipe version: {mp.__version__ if hasattr(mp, '__version__') else 'N/A'}")

print("\nInitializing MediaPipe Pose model...")
mp_pose = mp.solutions.pose.Pose()
print("MediaPipe Pose model loaded successfully!")

# Create dummy input (blank image)
print("\nCreating dummy test input...")
image = np.zeros((480, 640, 3), dtype=np.uint8)

print("Running inference on dummy input...")
results = mp_pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

if results.pose_landmarks:
    print("Pose landmarks detected.")
else:
    print("No landmarks detected (expected for blank image).")

print("\nMediaPipe environment is working correctly.")
print("Next steps:\n1. Test on real images using mediapipe_inference.py")
