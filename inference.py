import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import argparse
import os
import sys

# --- 1. Argument Parser ---
parser = argparse.ArgumentParser(description="Run Depth Estimation Inference")
parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
parser.add_argument("--weights", type=str, default="hybrid_best.pth", help="Path to model weights")
parser.add_argument("--backbone", type=str, default="dinov3_vitl16.pth", help="Path to backbone weights")
args = parser.parse_args()

# --- 2. Device Setup ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on: {device}")

# --- 3. Robust Model Definition (Fixing Paths) ---
# We redefine the wrapper class here to avoid 'da3_adapter.py' hardcoded path errors
try:
    from da3_adapter import DPTHead
except ImportError:
    print("Error: Could not import DPTHead from da3_adapter.py.")
    print("Make sure you are in the project root.")
    sys.exit(1)

class RobustHybridModel(nn.Module):
    def __init__(self, backbone_path):
        super().__init__()
        # Force local loading of DINOv3 to avoid 403 Forbidden errors
        # We assume 'dinov3' folder exists in current directory
        if not os.path.exists("dinov3"):
            print("Error: 'dinov3' folder not found. Please clone it first.")
            sys.exit(1)

        print("Loading DINOv3 backbone locally...")
        self.backbone = torch.hub.load('./dinov3', 'dinov3_vitl16', source='local', pretrained=False)
        self.head = DPTHead(in_channels=[1024, 1024, 1024, 1024])

    def forward(self, x):
        all_feat = self.backbone.get_intermediate_layers(x, n=24, reshape=True)
        features = [all_feat[4], all_feat[11], all_feat[17], all_feat[23]]
        return self.head(features)

# --- 4. Load Model ---
if not os.path.exists(args.backbone):
    print(f"Error: Backbone weights '{args.backbone}' not found.")
    print("Please download them using: huggingface-cli download Junming111/task3 dinov3_vitl16.pth --local-dir .")
    sys.exit(1)

print("Initializing model...")
model = RobustHybridModel(args.backbone).to(device)

# Load Backbone Weights
state_backbone = torch.load(args.backbone, map_location=device)
model.backbone.load_state_dict(state_backbone, strict=False)

# Load Adapter Weights
if os.path.exists(args.weights):
    print(f"Loading adapter weights: {args.weights}")
    state_adapter = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_adapter, strict=False)
else:
    print(f"Warning: Adapter weights '{args.weights}' not found. Using random weights for adapter.")

model.eval()

# --- 5. Run Inference ---
if not os.path.exists(args.image_path):
    print(f"Error: Image '{args.image_path}' not found.")
    sys.exit(1)

img = Image.open(args.image_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

input_tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    prediction = model(input_tensor)

# --- 6. Visualize ---
prediction = prediction.squeeze().cpu().numpy()

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(img.resize((512, 512)))
ax[0].set_title("Input Image")
ax[0].axis('off')

ax[1].imshow(prediction, cmap='magma')
ax[1].set_title("Depth Prediction")
ax[1].axis('off')

plt.tight_layout()
output_file = "prediction_result.png"
plt.savefig(output_file)
print(f"Result saved to {output_file}")
plt.show()