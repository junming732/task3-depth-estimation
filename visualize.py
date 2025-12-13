import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import os
import argparse
import glob
from PIL import Image
from dataset_universal import UniversalDepthDataset
from model_phase1 import DINOv3LinearProbe
try:
    from da3_adapter import DINOv3_DA3_Hybrid
except ImportError:
    DINOv3_DA3_Hybrid = None

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, choices=['test_set', 'custom'])
parser.add_argument('--model_type', type=str, required=True, choices=['probe', 'da3'])
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--input_folder', type=str, default='my_custom_images')
args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Setup Model
if args.model_type == 'probe':
    input_size = (224, 224)
    model = DINOv3LinearProbe(output_size=input_size).to(DEVICE)
elif args.model_type == 'da3':
    input_size = (518, 518)
    model = DINOv3_DA3_Hybrid().to(DEVICE)

state_dict = torch.load(args.checkpoint, map_location=DEVICE)
clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(clean_state_dict, strict=False) # strict=False is safer for viz
model.eval()

viz_transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def colorize(tensor, cmap='magma'):
    tensor = tensor.cpu().numpy()
    vmin, vmax = np.percentile(tensor, 2), np.percentile(tensor, 98)
    tensor = np.clip(tensor, vmin, vmax)
    norm = (tensor - vmin) / (vmax - vmin + 1e-5)
    return plt.get_cmap(cmap)(norm)[:, :, :3]

if args.mode == 'test_set':
    print("Visualizing NYU Test Set...")
    dataset = UniversalDepthDataset('nyu', 'test', transform=viz_transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    images, gt_depth = batch['image'].to(DEVICE), batch['depth'].to(DEVICE)

    with torch.no_grad():
        pred = model(images)
        pred = torch.exp(pred) # <--- CRITICAL FIX: UN-LOG

        gt_resized = F.interpolate(gt_depth, size=(224, 224), mode='bilinear')
        pred_resized = F.interpolate(pred, size=(224, 224), mode='bilinear')

    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    plt.suptitle(f"Results: {args.model_type.upper()} on NYU", fontsize=16)

    for i in range(4):
        # RGB
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        axes[i, 0].imshow(np.clip(img, 0, 1))
        axes[i, 0].set_title("Input")

        # GT
        axes[i, 1].imshow(colorize(gt_resized[i].squeeze()))
        axes[i, 1].set_title("Ground Truth")

        # Pred
        axes[i, 2].imshow(colorize(pred_resized[i].squeeze()))
        axes[i, 2].set_title("Prediction")

        for ax in axes[i]: ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'viz_nyu_{args.model_type}.png')
    print(f"Saved viz_nyu_{args.model_type}.png")

elif args.mode == 'custom':
    print(f"Visualizing Custom Images from {args.input_folder}...")
    image_paths = sorted(glob.glob(os.path.join(args.input_folder, "*.[jJ][pP][gG]")))

    if not image_paths:
        print(f"ERROR: No images found in {args.input_folder}.")
        print("Please upload .jpg files (e.g., photo of your room) to this folder!")
        exit()

    for img_path in image_paths:
        name = os.path.basename(img_path)
        pil_img = Image.open(img_path).convert('RGB')
        input_tensor = viz_transform(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = model(input_tensor)
            pred = torch.exp(pred) # <--- CRITICAL FIX: UN-LOG

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(pil_img.resize((224, 224)))
        axes[0].set_title("Input")
        axes[0].axis('off')

        axes[1].imshow(colorize(pred[0].squeeze()))
        axes[1].set_title("Predicted Depth")
        axes[1].axis('off')

        out_name = f'viz_custom_{name}_{args.model_type}.png'
        plt.savefig(out_name)
        plt.close()
        print(f"Saved {out_name}")