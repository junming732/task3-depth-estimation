import torch
import argparse
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

# --- IMPORTS ---
from dataset_universal import UniversalDepthDataset
from model_phase1 import DINOv3LinearProbe
try:
    from da3_adapter import DINOv3_DA3_Hybrid
except ImportError:
    DINOv3_DA3_Hybrid = None

# --- ARGS ---
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, required=True, choices=['probe', 'da3'])
parser.add_argument('--dataset', type=str, required=True, choices=['nyu', 'eth3d'])
parser.add_argument('--checkpoint', type=str, required=True)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL SETUP ---
if args.model_type == 'probe':
    input_size = (224, 224)
    model = DINOv3LinearProbe(output_size=input_size).to(device)
elif args.model_type == 'da3':
    input_size = (518, 518)
    model = DINOv3_DA3_Hybrid().to(device)

print(f"Loading {args.dataset.upper()} Test Set...")
eval_transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = UniversalDepthDataset(dataset_name=args.dataset, split='test', transform=eval_transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

print(f"Loading weights from {args.checkpoint}...")
state_dict = torch.load(args.checkpoint, map_location=device)
clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(clean_state_dict, strict=True)
model.eval()

# --- METRICS ---
def compute_metrics(pred, target):
    # 1. Un-Log (Both Phase 1 and Phase 3 were trained on Log Depth)
    pred = torch.exp(pred)

    # 2. Resize to GT
    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)

    # 3. Mask valid pixels
    mask = (target > 1e-3) & (target < 10) & (pred > 1e-3)
    pred = pred[mask]
    target = target[mask]

    if len(target) == 0: return 0.0, 0.0, 0.0

    # 4. MEDIAN SCALING (Crucial for SI Loss models)
    # We align the median of the prediction to the median of the target
    scale = torch.median(target) / torch.median(pred)
    pred = pred * scale

    # 5. Metrics
    rmse = torch.sqrt(torch.mean((target - pred) ** 2))
    abs_rel = torch.mean(torch.abs(target - pred) / target)
    thresh = torch.max((target / pred), (pred / target))
    a1 = (thresh < 1.25).float().mean()

    return rmse.item(), abs_rel.item(), a1.item()

print(f"Starting Evaluation on {len(test_dataset)} images...")
total_rmse, total_abs_rel, total_a1, count = 0.0, 0.0, 0.0, 0

with torch.no_grad():
    for i, batch in enumerate(test_loader):
        img = batch['image'].to(device)
        depth = batch['depth'].to(device)

        pred = model(img)
        rmse, abs_rel, a1 = compute_metrics(pred, depth)

        batch_n = len(img)
        total_rmse += rmse * batch_n
        total_abs_rel += abs_rel * batch_n
        total_a1 += a1 * batch_n
        count += batch_n

        if i % 10 == 0:
            print(f"Batch {i}: RMSE={rmse:.4f}, Acc={a1:.4f}")

print("\n" + "=" * 40)
print(f"RESULTS: {args.model_type.upper()} on {args.dataset.upper()}")
print(f"RMSE (Lower is better):    {total_rmse/count:.4f}")
print(f"AbsRel (Lower is better):  {total_abs_rel/count:.4f}")
print(f"Accuracy (Higher is better): {total_a1/count:.4f}")
print("=" * 40)