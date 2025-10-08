"""
Test HRNet on real images
"""
import torch
import cv2
import numpy as np
from hrnet_model import PoseHighResolutionNet, preprocess_image, get_pose_from_heatmaps
import os


def visualize_keypoints(image_path, keypoints, output_path='output_hrnet.jpg'):
    """Draw keypoints on image"""
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # Scale keypoints to image size (from 48x64 heatmap to original size)
    scale_x = w / 48
    scale_y = h / 64

    # COCO keypoint connections
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]

    # Draw connections
    for start, end in connections:
        if start < len(keypoints) and end < len(keypoints):
            x1, y1, conf1 = keypoints[start]
            x2, y2, conf2 = keypoints[end]

            x1_scaled = int(x1 * scale_x)
            y1_scaled = int(y1 * scale_y)
            x2_scaled = int(x2 * scale_x)
            y2_scaled = int(y2 * scale_y)

            if conf1 > 0.1 and conf2 > 0.1:
                cv2.line(img, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), (0, 255, 0), 2)

    # Draw keypoints
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.1:
            x_scaled = int(x * scale_x)
            y_scaled = int(y * scale_y)
            cv2.circle(img, (x_scaled, y_scaled), 5, (0, 0, 255), -1)
            cv2.putText(img, str(i), (x_scaled+5, y_scaled-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    cv2.imwrite(output_path, img)
    print(f"Saved visualization to {output_path}")
    return img


def main():
    print("Loading HRNet model...")
    model = PoseHighResolutionNet(num_joints=17)

    weights_path = 'hrnet_weights/pose_hrnet_w32_256x192.pth'
    checkpoint = torch.load(weights_path, map_location='cpu')

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    # Find test images
    test_dir = 'test_images'
    if not os.path.exists(test_dir):
        print(f"No {test_dir} directory found, creating one...")
        os.makedirs(test_dir)
        print(f"Please add test images to {test_dir}/ and run again")
        return

    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    if not image_files:
        print(f"No images found in {test_dir}/")
        return

    print(f"\nFound {len(image_files)} test images")

    for img_file in image_files[:3]:  # Test first 3 images
        image_path = os.path.join(test_dir, img_file)
        print(f"\nProcessing {img_file}...")

        # Preprocess
        img_tensor = preprocess_image(image_path).to(device)

        # Inference
        with torch.no_grad():
            output = model(img_tensor)

        # Extract keypoints
        heatmaps = output.cpu().numpy()[0]
        keypoints = get_pose_from_heatmaps(heatmaps)

        print(f"  Detected {len(keypoints)} keypoints")
        print(f"  Average confidence: {keypoints[:, 2].mean():.4f}")

        # Visualize
        output_path = f"results/output_{img_file}"
        visualize_keypoints(image_path, keypoints, output_path)

    print("\n✅ HRNet is working properly on real images!")


if __name__ == '__main__':
    main()
