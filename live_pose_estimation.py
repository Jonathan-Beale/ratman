"""
Live Pose Estimation using rtmlib (DWPose/RTMPose via DirectML)
Displays real-time whole-body pose estimation from webcam.

Backend: onnxruntime-directml for GPU acceleration on Windows.
Models are downloaded automatically to ~/.cache/rtmlib/ on first run.
"""

import cv2
import numpy as np
import time

# Patch rtmlib to support DirectML (Windows GPU via DirectX)
from rtmlib.tools.base import RTMLIB_SETTINGS
RTMLIB_SETTINGS['onnxruntime']['dml'] = 'DmlExecutionProvider'

from rtmlib import Wholebody, draw_skeleton
import onnxruntime as ort


def get_best_device():
    providers = ort.get_available_providers()
    if 'DmlExecutionProvider' in providers:
        return 'dml'
    return 'cpu'


def main():
    device = get_best_device()
    print(f"Using device: {device}")

    print("Loading pose estimator (downloads models on first run)...")
    pose = Wholebody(
        mode='performance',
        to_openpose=True,
        backend='onnxruntime',
        device=device,
    )
    print("Loaded.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting live pose estimation. Press 'q' to quit, 'f' to toggle fullscreen.")

    WINDOW = "Live Pose Estimation"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    fps_counter = 0
    fps_display = 0.0
    fps_timer = time.time()
    fullscreen = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break

        frame = cv2.flip(frame, 1)

        keypoints, scores = pose(frame)

        annotated = draw_skeleton(frame, keypoints, scores, openpose_skeleton=True, kpt_thr=0.3)

        # FPS
        fps_counter += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            fps_timer = time.time()

        cv2.putText(annotated, f"FPS: {fps_display:.1f}  device: {device}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, "f: fullscreen  q: quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(WINDOW, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("f"):
            fullscreen = not fullscreen
            prop = cv2.WND_PROP_FULLSCREEN
            cv2.setWindowProperty(WINDOW, prop,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

    cap.release()
    cv2.destroyAllWindows()
    print("Pose estimation closed.")


if __name__ == "__main__":
    main()
