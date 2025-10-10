import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import os 
import glob 
import random

# COCO class names
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def get_model(num_classes):
    """Load the model architecture"""
    model = maskrcnn_resnet50_fpn(weights=None)
    
    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    return model

def load_model(model_path, num_classes=91, device='cpu'):
    """Load trained model weights"""
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict(model, image_path, device='cpu', threshold=0.5):
    """Run inference on an image"""
    # Load and transform image
    img = Image.open(image_path).convert('RGB')
    transform = T.ToTensor()
    img_tensor = transform(img).to(device)
    
    # Run prediction
    with torch.no_grad():
        predictions = model([img_tensor])[0]
    
    # Filter by confidence threshold
    keep = predictions['scores'] > threshold
    boxes = predictions['boxes'][keep].cpu().numpy()
    labels = predictions['labels'][keep].cpu().numpy()
    scores = predictions['scores'][keep].cpu().numpy()
    masks = predictions['masks'][keep].cpu().numpy()
    
    return img, boxes, labels, scores, masks

def visualize_prediction(img, boxes, labels, scores, masks, output_path, threshold=0.5):
    """Visualize predictions with boxes and masks and save to output_path."""
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img)
    
    # Create color map for masks
    if len(boxes) > 0:
        colors = plt.cm.hsv(np.linspace(0, 1, len(boxes))).tolist()
    else:
        colors = []
    
    for i, (box, label, score, mask) in enumerate(zip(boxes, labels, scores, masks)):
        if score < threshold:
            continue
            
        # Draw bounding box
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        rect = plt.Rectangle((x1, y1), width, height, 
                             fill=False, edgecolor=colors[i], linewidth=2)
        ax.add_patch(rect)
        
        # Draw mask
        mask = mask[0] > 0.5  # Binary threshold
        colored_mask = np.zeros((*mask.shape, 4))
        colored_mask[mask] = colors[i]
        colored_mask[mask, 3] = 0.5  # Alpha
        ax.imshow(colored_mask)
        
        # Add label
        class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"Class {label}"
        text = f"{class_name}: {score:.2f}"
        ax.text(x1, y1 - 5, text, color='white', fontsize=10,
                bbox=dict(facecolor=colors[i], alpha=0.7, edgecolor='none', pad=1))
    
    ax.axis('off')
    plt.tight_layout()
    # Save the figure instead of showing it
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig) # Close the figure to free up memory

def main():
    # Configuration
    model_path = 'maskrcnn_coco_model.pth'
    input_dir = 'coco_data/val2017'
    output_dir = 'inference'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    confidence_threshold = 0.5
    max_inferences = 100


    print(f"Using device: {device}")
    
    # --- Directory Setup ---
    # 1. Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Inferences will be saved to: {output_dir}")
    
    # 2. Find all jpg images in the input directory
    # Use glob.glob to find all files ending with .jpg (case-insensitive)
    image_paths = glob.glob(os.path.join(input_dir, '*.jpg'))
    if not image_paths:
        print(f"No .jpg files found in {input_dir}. Exiting.")
        return
    print(f"Found {len(image_paths)} images to process.")

    if max_inferences > 0 and len(image_paths) > max_inferences:
        # 1. Randomly shuffle the entire list of paths
        random.shuffle(image_paths) 
        
        # 2. Select the first N paths from the now-shuffled list
        image_paths = image_paths[:max_inferences] 
        print(f"Randomly selected {max_inferences} images for quick testing.")
    
    # --- Model Loading ---
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, num_classes=91, device=device)
    print("Model loaded successfully!")
    
    # --- Batch Inference ---
    for i, image_path in enumerate(image_paths):
        print(f"\nProcessing image {i+1}/{len(image_paths)}: {image_path}...")
        
        try:
            # 1. Run prediction
            img, boxes, labels, scores, masks = predict(model, image_path, device, confidence_threshold)
            
            print(f"Found {len(boxes)} objects in {os.path.basename(image_path)}.")
            
            # 2. Determine output path
            file_name = os.path.basename(image_path)
            output_path = os.path.join(output_dir, file_name.replace('.jpg', '_inference.png'))
            
            # 3. Visualize and save
            visualize_prediction(img, boxes, labels, scores, masks, output_path, confidence_threshold)
            print(f"Inference saved to {output_path}")

        except Exception as e:
            print(f"An error occurred while processing {image_path}: {e}")
            continue # Continue to the next image if one fails

if __name__ == '__main__':
    main()