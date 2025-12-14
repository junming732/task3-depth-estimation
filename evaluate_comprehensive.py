import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random
import sys

# --- IMPORTS (Adjust if your folder structure is different) ---
sys.path.append(os.getcwd())
from dataset_universal import UniversalDepthDataset
from da3_adapter import DINOv3_DA3_Hybrid
from model_phase1 import DINOv3LinearProbe
from tiny_unet import UNet

# --- CONFIGURATION ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = "results_comprehensive"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Checkpoints (From your ls -lh)
CKPT_PHASE3 = "checkpoints_phase3_nyu/hybrid_best.pth"
CKPT_PHASE1 = "checkpoints_phase1_nyu/model_epoch_15.pth"
CKPT_UNET   = "models/unet_highres/model_10.pth"

# --- HELPER FUNCTIONS ---

def compute_errors(gt, pred):
    """Compute metrics for a single batch or image."""
    # Align dimensions
    if pred.shape != gt.shape:
        pred = F.interpolate(pred, size=gt.shape[-2:], mode='bilinear', align_corners=False)

    # Valid mask
    mask = gt > 0.001
    pred_valid = pred[mask]
    gt_valid = gt[mask]

    if len(gt_valid) == 0:
        return None

    # Median Scaling (Crucial for Monocular Comparison)
    scale = torch.median(gt_valid) / (torch.median(pred_valid) + 1e-8)
    pred_valid = pred_valid * scale

    # Metrics
    thresh = torch.max((gt_valid / pred_valid), (pred_valid / gt_valid))
    a1 = (thresh < 1.25).float().mean()

    rmse = (gt_valid - pred_valid) ** 2
    rmse = torch.sqrt(rmse.mean())

    abs_rel = torch.mean(torch.abs(gt_valid - pred_valid) / gt_valid)

    return {'a1': a1.item(), 'rmse': rmse.item(), 'abs_rel': abs_rel.item(), 'scale': scale.item()}

def load_model(model_type, checkpoint_path):
    print(f"Loading {model_type} from {checkpoint_path}...")
    if model_type == 'phase3':
        model = DINOv3_DA3_Hybrid()
    elif model_type == 'phase1':
        model = DINOv3LinearProbe(output_size=(224, 224))
    elif model_type == 'unet':
        model = UNet()

    # Load weights
    try:
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"Warning: strict loading failed for {model_type}, trying loose match. {e}")
        # Fallback for keys with module. prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v
        model.load_state_dict(new_state_dict, strict=False)

    model.to(DEVICE)
    model.eval()
    return model

# --- MAIN EVALUATION LOGIC ---

def evaluate_dataset(model, dataloader, transform_size):
    """Runs inference on entire dataset and returns metrics + per-image RMSE."""
    metrics_sum = {'a1': 0, 'rmse': 0, 'abs_rel': 0}
    image_scores = [] # (index, rmse)
    count = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            img = batch['image'].to(DEVICE)
            depth_gt = batch['depth'].to(DEVICE)

            # Resize input if needed
            if img.shape[-1] != transform_size:
                img = F.interpolate(img, size=(transform_size, transform_size), mode='bilinear', align_corners=False)

            # Inference
            pred = model(img)

            # Compute Error
            err = compute_errors(depth_gt, pred)
            if err:
                metrics_sum['a1'] += err['a1']
                metrics_sum['rmse'] += err['rmse']
                metrics_sum['abs_rel'] += err['abs_rel']
                image_scores.append((i, err['rmse'])) # Store RMSE to find best/worst
                count += 1

    # Average
    avg_metrics = {k: v/count for k, v in metrics_sum.items()}
    return avg_metrics, image_scores

def get_prediction(model, dataset, index, transform_size):
    """Get visualization data for a specific image index."""
    batch = dataset[index]
    img_tensor = batch['image'].unsqueeze(0).to(DEVICE)
    gt_tensor = batch['depth'].unsqueeze(0).to(DEVICE)

    # Resize input
    if img_tensor.shape[-1] != transform_size:
        img_tensor = F.interpolate(img_tensor, size=(transform_size, transform_size), mode='bilinear', align_corners=False)

    with torch.no_grad():
        pred = model(img_tensor)

    # Resize pred to GT for visualization
    pred = F.interpolate(pred, size=gt_tensor.shape[-2:], mode='bilinear', align_corners=False)

    # Get Scale for visualization consistency
    # We cheat slightly for viz by median-scaling pred to GT so colors match
    valid_mask = gt_tensor > 0.001
    scale = torch.median(gt_tensor[valid_mask]) / (torch.median(pred[valid_mask]) + 1e-8)
    pred_scaled = pred * scale

    return pred_scaled.squeeze().cpu().numpy()

# --- EXECUTION ---

def main():
    # 1. Dataset Setup (Use Validation Set)
    print("Setting up Dataset...")
    # Base transform (High Res)
    transform_base = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = UniversalDepthDataset('nyu', 'val', transform=transform_base)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # 2. Evaluate Phase 3 (The "Hero" Model)
    print("\n--- Evaluating Phase 3 (Hybrid) ---")
    model_p3 = load_model('phase3', CKPT_PHASE3)
    metrics_p3, scores_p3 = evaluate_dataset(model_p3, val_loader, 512)
    print(f"Phase 3 Results: {metrics_p3}")

    # Find Best, Worst, Random
    scores_p3.sort(key=lambda x: x[1]) # Sort by RMSE
    best_idx = scores_p3[0][0]
    worst_idx = scores_p3[-1][0]
    random_idx = random.choice(scores_p3)[0]

    selected_indices = {'Best': best_idx, 'Worst': worst_idx, 'Random': random_idx}
    print(f"Selected Indices: {selected_indices}")

    # Store predictions for qualitative plot
    viz_data = {k: {} for k in selected_indices}

    # Capture Phase 3 Predictions
    for case, idx in selected_indices.items():
        viz_data[case]['phase3'] = get_prediction(model_p3, val_dataset, idx, 512)
        # Also grab GT and RGB
        batch = val_dataset[idx]
        viz_data[case]['gt'] = batch['depth'].squeeze().numpy()
        # Un-normalize RGB for display
        rgb = batch['image'].permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        rgb = std * rgb + mean
        viz_data[case]['rgb'] = np.clip(rgb, 0, 1)

    del model_p3 # Save memory
    torch.cuda.empty_cache()

    # 3. Evaluate Phase 1
    print("\n--- Evaluating Phase 1 (Probe) ---")
    model_p1 = load_model('phase1', CKPT_PHASE1)
    metrics_p1, _ = evaluate_dataset(model_p1, val_loader, 224) # Phase 1 used 224
    print(f"Phase 1 Results: {metrics_p1}")

    for case, idx in selected_indices.items():
        viz_data[case]['phase1'] = get_prediction(model_p1, val_dataset, idx, 224)

    del model_p1
    torch.cuda.empty_cache()

    # 4. Evaluate UNet
    print("\n--- Evaluating UNet (Baseline) ---")
    model_unet = load_model('unet', CKPT_UNET)
    metrics_unet, _ = evaluate_dataset(model_unet, val_loader, 224) # Assume 224 for Unet
    print(f"UNet Results: {metrics_unet}")

    for case, idx in selected_indices.items():
        viz_data[case]['unet'] = get_prediction(model_unet, val_dataset, idx, 224)

    del model_unet

    # --- REPORTING ---

    # A. Quantitative Table
    print("\n" + "="*60)
    print(f"{'Model':<20} | {'RMSE (Lower better)':<20} | {'AbsRel':<10} | {'Acc < 1.25':<10}")
    print("-" * 60)
    print(f"{'UNet (Baseline)':<20} | {metrics_unet['rmse']:.4f}{' ':14} | {metrics_unet['abs_rel']:.4f}     | {metrics_unet['a1']:.4f}")
    print(f"{'Phase 1 (Probe)':<20} | {metrics_p1['rmse']:.4f}{' ':14} | {metrics_p1['abs_rel']:.4f}     | {metrics_p1['a1']:.4f}")
    print(f"{'Phase 3 (Hybrid)':<20} | {metrics_p3['rmse']:.4f}{' ':14} | {metrics_p3['abs_rel']:.4f}     | {metrics_p3['a1']:.4f}")
    print("="*60)

    # B. Qualitative Plot
    print(f"\nGenerating Comparison Plot in {OUTPUT_DIR}...")
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    rows = ['Best', 'Worst', 'Random']
    cols = ['RGB', 'Ground Truth', 'UNet', 'Phase 1', 'Phase 3 (Ours)']

    for i, row_name in enumerate(rows):
        data = viz_data[row_name]

        # Plot RGB
        axes[i, 0].imshow(data['rgb'])
        axes[i, 0].set_title(f"{row_name}\nRGB")
        axes[i, 0].axis('off')

        # Plot Depths (Use common vmin/vmax from GT for fairness)
        vmin, vmax = data['gt'].min(), data['gt'].max()

        im1 = axes[i, 1].imshow(data['gt'], cmap='magma', vmin=vmin, vmax=vmax)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')

        im2 = axes[i, 2].imshow(data['unet'], cmap='magma', vmin=vmin, vmax=vmax)
        axes[i, 2].set_title("UNet")
        axes[i, 2].axis('off')

        im3 = axes[i, 3].imshow(data['phase1'], cmap='magma', vmin=vmin, vmax=vmax)
        axes[i, 3].set_title("Phase 1 (Probe)")
        axes[i, 3].axis('off')

        im4 = axes[i, 4].imshow(data['phase3'], cmap='magma', vmin=vmin, vmax=vmax)
        axes[i, 4].set_title("Phase 3 (Hybrid)")
        axes[i, 4].axis('off')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/comparison_comprehensive.png")
    print("Done.")

if __name__ == "__main__":
    main()