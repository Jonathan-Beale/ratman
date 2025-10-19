"""
Test HRNet on real images using official code and visualize results
"""
import torch
import cv2
import numpy as np
import yaml
from hrnet_model import get_pose_net
import os


def preprocess_image(image_path, input_size=(192, 256)):
    """Preprocess image for HRNet"""
    img = cv2.imread(image_path)
    original_img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_size)
    img = img.astype(np.float32) / 255.0
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    return img, original_img


def get_pose_from_heatmaps(heatmaps):
    """Extract keypoint coordinates from heatmaps"""
    keypoints = []
    for i in range(heatmaps.shape[0]):
        heatmap = heatmaps[i]
        idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        confidence = heatmap[idx]
        keypoints.append([idx[1], idx[0], confidence])
    return np.array(keypoints)


def visualize_keypoints(image, keypoints, output_path):
    """Draw keypoints on image"""
    img = image.copy()
    h, w = img.shape[:2]

    # Scale keypoints from heatmap size (48x64) to original image size
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

            if conf1 > 0.3 and conf2 > 0.3:  # Higher confidence threshold
                x1_scaled = int(x1 * scale_x)
                y1_scaled = int(y1 * scale_y)
                x2_scaled = int(x2 * scale_x)
                y2_scaled = int(y2 * scale_y)
                cv2.line(img, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), (0, 255, 0), 3)

    # Draw keypoints
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.3:  # Only draw high confidence keypoints
            x_scaled = int(x * scale_x)
            y_scaled = int(y * scale_y)
            cv2.circle(img, (x_scaled, y_scaled), 8, (0, 0, 255), -1)
            cv2.putText(img, f'{conf:.2f}', (x_scaled+10, y_scaled-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    cv2.imwrite(output_path, img)
    print(f"✅ Saved visualization to {output_path}")


def main():
    print("="*60)
    print("TESTING OFFICIAL HRNET WITH REAL IMAGES")
    print("="*60)

    # Load config and model
    with open('hrnet_config.yaml') as f:
        cfg = yaml.safe_load(f)

    model = get_pose_net(cfg, is_train=False)
    checkpoint = torch.load('hrnet_weights/pose_hrnet_w32_256x192_official.pth', map_location='cpu')
    model.load_state_dict(checkpoint, strict=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Test on COCO subset images
    test_dir = 'coco_subset/images'
    if not os.path.exists(test_dir):
        print(f"❌ Test directory not found: {test_dir}")
        print("Run create_coco_subset.py first!")
        return

    image_files = sorted([f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    print(f"\nFound {len(image_files)} test images\n")

    os.makedirs('output', exist_ok=True)

    for img_file in image_files:
        image_path = os.path.join(test_dir, img_file)
        print(f"\n{'='*60}")
        print(f"Processing: {img_file}")
        print(f"{'='*60}")

        img_tensor, original_img = preprocess_image(image_path)
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            output = model(img_tensor)

        heatmaps = output.cpu().numpy()[0]
        keypoints = get_pose_from_heatmaps(heatmaps)

        print(f"Detected {len(keypoints)} keypoints")
        print(f"Average confidence: {keypoints[:, 2].mean():.4f}")
        print(f"Max confidence: {keypoints[:, 2].max():.4f}")
        print(f"High confidence keypoints (>0.5): {(keypoints[:, 2] > 0.5).sum()}")

        output_path = f"output/{img_file}"
        visualize_keypoints(original_img, keypoints, output_path)

    print(f"\n{'='*60}")
    print("✅ HR NET PREDICTIONS COMPLETE!")
    print(f"Ground truth (COCO annotations): ground_truth/")
    print(f"Model predictions: output/")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
