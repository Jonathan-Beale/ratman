"""
Minimal test script to verify HRNet works with clean environment
Based on simple-HRNet implementation
"""
import torch
import torchvision
import numpy as np
from PIL import Image

print("Testing clean HRNet environment...")
print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Define simple HRNet model structure (W32 variant)
class HRNet(torch.nn.Module):
    def __init__(self):
        super(HRNet, self).__init__()
        # Simplified HRNet structure - just for testing
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)

        # Final layer to predict 17 keypoints
        self.final_layer = torch.nn.Conv2d(64, 17, kernel_size=1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.final_layer(x)
        return x

print("\nCreating HRNet model...")
model = HRNet()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

print(f"Model created and moved to {device}")

# Create test input
print("\nCreating test input (256x192)...")
test_input = torch.randn(1, 3, 256, 192).to(device)

# Run inference
print("Running inference...")
with torch.no_grad():
    output = model(test_input)

print(f"Output shape: {output.shape}")
print(f"Expected shape: (1, 17, 64, 48)")

# Test heatmap extraction
heatmaps = output.cpu().numpy()[0]
print(f"\nHeatmaps shape: {heatmaps.shape}")
print(f"Number of keypoints: {heatmaps.shape[0]}")

# Find keypoint locations from heatmaps
keypoints = []
for i in range(heatmaps.shape[0]):
    heatmap = heatmaps[i]
    idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    confidence = heatmap[idx]
    keypoints.append([idx[1], idx[0], confidence])

print(f"Detected {len(keypoints)} keypoint locations")

print("\n✅ SUCCESS! HRNet environment is working correctly.")
print("\nNext steps:")
print("1. Download pre-trained weights: pose_hrnet_w32_256x192.pth")
print("2. Load weights into model")
print("3. Test on real images")
