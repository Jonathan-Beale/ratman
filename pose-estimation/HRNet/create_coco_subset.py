"""
Create a subset of 15 COCO validation images with ground truth annotations
"""
import json
import cv2
import numpy as np
import os
import urllib.request
from tqdm import tqdm

def visualize_coco_keypoints(image, keypoints, output_path):
    """Draw COCO ground truth keypoints"""
    img = image.copy()

    # COCO keypoint connections
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]

    # Keypoints: [x1,y1,v1,x2,y2,v2,...] where v: 0=not labeled, 1=labeled but not visible, 2=labeled and visible
    kpts = np.array(keypoints).reshape(-1, 3)

    # Draw connections
    for start, end in connections:
        if start < len(kpts) and end < len(kpts):
            x1, y1, v1 = kpts[start]
            x2, y2, v2 = kpts[end]
            if v1 > 0 and v2 > 0:
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

    # Draw keypoints
    for i, (x, y, v) in enumerate(kpts):
        if v > 0:  # Visible or occluded but labeled
            color = (0, 0, 255) if v == 2 else (255, 165, 0)  # Red if visible, orange if occluded
            cv2.circle(img, (int(x), int(y)), 8, color, -1)
            cv2.putText(img, 'GT', (int(x)+10, int(y)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    cv2.imwrite(output_path, img)
    print(f"  ✅ Saved ground truth: {output_path}")


def main():
    print("=" * 60)
    print("CREATING COCO SUBSET WITH GROUND TRUTH")
    print("=" * 60)

    # Load COCO annotations
    ann_file = 'coco_subset/person_keypoints_val2017.json'
    print(f"\nLoading {ann_file}...")

    with open(ann_file, 'r') as f:
        coco_data = json.load(f)

    print(f"Found {len(coco_data['images'])} images")
    print(f"Found {len(coco_data['annotations'])} annotations")

    # Build image_id to annotations mapping
    img_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    # Find images with good single-person keypoint annotations
    good_images = []
    for img_info in coco_data['images']:
        img_id = img_info['id']
        if img_id in img_to_anns:
            anns = img_to_anns[img_id]
            # Find best annotation (most visible keypoints)
            best_ann = None
            max_visible = 0
            for ann in anns:
                if 'keypoints' in ann and len(ann['keypoints']) == 51:  # 17 keypoints * 3
                    kpts = np.array(ann['keypoints']).reshape(-1, 3)
                    visible = np.sum(kpts[:, 2] == 2)  # Count visible keypoints
                    if visible > max_visible:
                        max_visible = visible
                        best_ann = ann

            if best_ann and max_visible >= 12:  # At least 12 visible keypoints
                good_images.append({
                    'image_info': img_info,
                    'annotation': best_ann,
                    'visible_kpts': max_visible
                })

    # Sort by number of visible keypoints and take top 15
    good_images.sort(key=lambda x: x['visible_kpts'], reverse=True)
    selected = good_images[:15]

    print(f"\n✅ Selected 15 images with best annotations")
    print(f"Visible keypoints range: {selected[-1]['visible_kpts']} to {selected[0]['visible_kpts']}")

    # Download images and create ground truth visualizations
    os.makedirs('coco_subset/images', exist_ok=True)
    os.makedirs('ground_truth', exist_ok=True)

    print("\nDownloading images and creating ground truth...")
    for idx, item in enumerate(selected, 1):
        img_info = item['image_info']
        ann = item['annotation']

        filename = f"image_{idx:03d}.jpg"
        img_path = f"coco_subset/images/{filename}"
        gt_path = f"ground_truth/{filename}"

        # Download image
        if not os.path.exists(img_path):
            print(f"\n[{idx}/15] Downloading {img_info['file_name']}...")
            try:
                urllib.request.urlretrieve(img_info['coco_url'], img_path)
            except Exception as e:
                print(f"  ❌ Download failed: {e}")
                continue

        # Load image and create ground truth visualization
        image = cv2.imread(img_path)
        if image is None:
            print(f"  ❌ Failed to load {img_path}")
            continue

        visualize_coco_keypoints(image, ann['keypoints'], gt_path)
        print(f"  Image: {img_path}")
        print(f"  Visible keypoints: {item['visible_kpts']}/17")

    print("\n" + "=" * 60)
    print("✅ COCO SUBSET CREATED")
    print("=" * 60)
    print(f"\nGround truth (with COCO annotations): ground_truth/")
    print(f"Raw images: coco_subset/images/")
    print(f"\nNext: Run HRNet predictions to generate output/")


if __name__ == '__main__':
    main()
