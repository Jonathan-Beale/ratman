import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import DataLoader, Subset
from pycocotools.coco import COCO
import torchvision.transforms as T
from PIL import Image
import os

class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms

    def __getitem__(self, idx):
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        path = coco.loadImgs(img_id)[0]['file_name']
        # Normalize path for Windows
        img_path = os.path.normpath(os.path.join(self.root, path))
        img = Image.open(img_path).convert('RGB')
        
        num_objs = len(anns)
        boxes = []
        labels = []
        masks = []
        
        for ann in anns:
            xmin, ymin, width, height = ann['bbox']
            boxes.append([xmin, ymin, xmin + width, ymin + height])
            labels.append(ann['category_id'])
            masks.append(coco.annToMask(ann))
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks
        target['image_id'] = torch.tensor([img_id])
        
        if self.transforms:
            img = self.transforms(img)
        else:
            img = T.ToTensor()(img)
        
        return img, target

    def __len__(self):
        return len(self.ids)

def get_model(num_classes):
    # Load pre-trained model
    model = maskrcnn_resnet50_fpn(weights='DEFAULT')
    
    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    return model

def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0
    
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
    
    return total_loss / len(data_loader)

def main():
    # Paths - UPDATE THESE
    train_img_dir = 'coco_data/val2017'
    train_ann_file = 'coco_data/annotations/instances_val2017.json'
    
    # Hyperparameters
    num_classes = 91  # COCO has 80 classes + background
    num_epochs = 5
    batch_size = 2
    learning_rate = 0.005
    num_samples = 100  # Use subset for quick testing
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset
    dataset = CocoDataset(train_img_dir, train_ann_file)
    dataset = Subset(dataset, range(num_samples))  # Use small subset
    
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=0  # Set to 0 for Windows
    )
    
    # Model
    model = get_model(num_classes)
    model.to(device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    
    # Training loop
    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(model, optimizer, data_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), 'maskrcnn_coco_model.pth')
    print("Training complete! Model saved.")

if __name__ == '__main__':
    main()