# This script processes images using MediaPipe Pose estimation.
# It reads images from a directory, runs pose estimation, and saves the annotated results.


import mediapipe as mp
import cv2
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def process_image(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Could not load {image_path}")
        return

    with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            print(f"No pose detected in {image_path}")
            return

        # Draw landmarks on the image
        annotated = image.copy()
        mp_drawing.draw_landmarks(
            annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imwrite(output_path, annotated)
        print(f"✅ Saved result to {output_path}")

def main():
    test_dir = "test_images"
    results_dir = "results"

    os.makedirs(results_dir, exist_ok=True)
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    print(f"Found {len(image_files)} images in {test_dir}")
    for img_file in image_files:
        input_path = os.path.join(test_dir, img_file)
        output_path = os.path.join(results_dir, f"output_{img_file}")
        process_image(input_path, output_path)

    print("\n✅ All test images processed successfully!")

if __name__ == "__main__":
    main()
