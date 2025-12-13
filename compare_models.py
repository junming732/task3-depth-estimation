import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import glob
import random

# --- IMPORTS ---
from tiny_unet import UNet
from model_phase1 import DINOv3LinearProbe
from da3_adapter import DINOv3_DA3_Hybrid
from dataset_universal import UniversalDepthDataset

# --- CONFIG ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_unet(path):
    model = UNet().to(DEVICE)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        print(f"[Phase 1] UNet loaded from {path}")
    else:
        print(f"[Phase 1] Warning: Checkpoint not found at {path}")
        return None
    model.eval()
    return model

def load_probe(path):
    model = DINOv3LinearProbe(output_size=(224, 224)).to(DEVICE)
    if os.path.exists(path):
        state_dict = torch.load(path, map_location=DEVICE)
        clean_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_dict, strict=False)
        print(f"[Phase 2] Linear Probe loaded from {path}")
    else:
        print(f"[Phase 2] Warning: Checkpoint not found at {path} (Skipping)")
        return None
    model.eval()
    return model

def load_hybrid(path):
    model = DINOv3_DA3_Hybrid().to(DEVICE)
    if os.path.exists(path):
        state_dict = torch.load(path, map_location=DEVICE)
        clean_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_dict, strict=True)
        print(f"[Phase 3] Hybrid DA3 loaded from {path}")
    else:
        print(f"[Phase 3] Warning: Checkpoint not found at {path}")
        return None
    model.eval()
    return model

def preprocess(pil_img, size):
    t = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return t(pil_img).unsqueeze(0).to(DEVICE)

def colorize(tensor, cmap='magma'):
    tensor = tensor.cpu().detach().numpy().squeeze()
    vmin, vmax = np.percentile(tensor, 2), np.percentile(tensor, 98)
    tensor = np.clip(tensor, vmin, vmax)
    norm = (tensor - vmin) / (vmax - vmin + 1e-5)
    return plt.get_cmap(cmap)(norm)[:, :, :3]

def run_comparison(pil_img, img_name, models, gt_depth=None):
    # Determine columns based on available models
    active_models = [k for k, v in models.items() if v is not None]

    # Base columns: Input + (Optional GT) + Active Models
    cols = 1 + (1 if gt_depth is not None else 0) + len(active_models)

    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 5))

    # Handle single axis case just in case
    if cols == 1: axes = [axes]

    current_col = 0

    # 1. Input Image
    axes[current_col].imshow(pil_img)
    axes[current_col].set_title("Input RGB")
    axes[current_col].axis('off')
    current_col += 1

    # Optional: Ground Truth
    if gt_depth is not None:
        axes[current_col].imshow(colorize(gt_depth))
        axes[current_col].set_title("Ground Truth")
        axes[current_col].axis('off')
        current_col += 1

    # 2. Phase 1: UNet
    if models['unet']:
        inp = preprocess(pil_img, (240, 320))
        with torch.no_grad():
            pred = models['unet'](inp)
            pred = torch.exp(pred)
        axes[current_col].imshow(colorize(pred))
        axes[current_col].set_title("Phase 1: Tiny UNet\n(Scratch)")
        axes[current_col].axis('off')
        current_col += 1

    # 3. Phase 2: Linear Probe
    if models['probe']:
        inp = preprocess(pil_img, (224, 224))
        with torch.no_grad():
            pred = models['probe'](inp)
            pred = torch.exp(pred)
        axes[current_col].imshow(colorize(pred))
        axes[current_col].set_title("Phase 2: Probe\n(Frozen)")
        axes[current_col].axis('off')
        current_col += 1

    # 4. Phase 3: Hybrid
    if models['hybrid']:
        inp = preprocess(pil_img, (512, 512))
        with torch.no_grad():
            pred = models['hybrid'](inp)
            pred = torch.exp(pred)
        axes[current_col].imshow(colorize(pred))
        axes[current_col].set_title("Phase 3: Hybrid\n(DINOv3 + Adapter)")
        axes[current_col].axis('off')
        current_col += 1

    plt.tight_layout()
    save_path = f"comparison_{img_name}.png"
    plt.savefig(save_path)
    print(f"Saved comparison to {save_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='nyu', choices=['custom', 'nyu'])
    parser.add_argument('--img_folder', type=str, default='my_custom_images')
    parser.add_argument('--samples', type=int, default=5)
    args = parser.parse_args()

    # --- STRICT PATH CONFIGURATION ---
    # Using the exact paths you provided
    unet_path = 'models/unet_highres/model_10.pth'
    probe_path = 'checkpoints_phase1_nyu/model_epoch_10.pth' # <--- Updated to your path
    hybrid_path = 'checkpoints_phase3_nyu/hybrid_best.pth'

    # Load Models
    models = {
        'unet': load_unet(unet_path),
        'probe': load_probe(probe_path),
        'hybrid': load_hybrid(hybrid_path)
    }

    if args.mode == 'custom':
        images = sorted(glob.glob(os.path.join(args.img_folder, "*.[jJ][pP][gG]")))
        if not images:
            print(f"No images found in {args.img_folder}")
        for i, img_path in enumerate(images):
            pil_img = Image.open(img_path).convert('RGB')
            run_comparison(pil_img, f"custom_{i}", models)

    elif args.mode == 'nyu':
        print(f"Loading NYU Test Set (Random {args.samples} samples)...")
        dataset = UniversalDepthDataset('nyu', 'test', transform=None)

        indices = random.sample(range(len(dataset)), args.samples)

        for i, idx in enumerate(indices):
            sample = dataset[idx]
            pil_img = sample['image']
            depth_tensor = sample['depth']

            run_comparison(pil_img, f"nyu_{i}", models, gt_depth=depth_tensor)