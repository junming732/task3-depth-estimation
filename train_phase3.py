import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import os
import argparse
import sys

# Ensure we can find the modules
sys.path.append(os.getcwd())

from dataset_universal import UniversalDepthDataset
from da3_adapter import DINOv3_DA3_Hybrid

# --- Args ---
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, choices=['nyu', 'eth3d'])
parser.add_argument('--epochs', type=int, default=30) # Default updated to 30
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--lr', type=float, default=5e-5)
args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = f'checkpoints_phase3_{args.dataset}'
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"--- Starting Phase 3 (Hybrid) on {args.dataset.upper()} ---")
print(f"--- Saving Mode: Minimal (Latest + Best only) ---")

# --- CORRECT TRANSFORM (512x512) ---
da3_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_loader = DataLoader(
    UniversalDepthDataset(args.dataset, 'train', transform=da3_transform),
    batch_size=args.batch_size, shuffle=True, num_workers=4
)

# Use a small subset for validation to save time
val_dataset = UniversalDepthDataset(args.dataset, 'val', transform=da3_transform)
val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size, shuffle=False, num_workers=4
)

model = DINOv3_DA3_Hybrid().to(DEVICE)

# Update Optimizer to include unfrozen backbone layers
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

def si_loss(pred, target):
    di = pred - target
    n = (di.shape[-1] * di.shape[-2])
    di2 = torch.pow(di, 2)
    first_term = torch.sum(di2, dim=(1,2,3)) / n
    second_term = 0.5 * (torch.pow(torch.sum(di, dim=(1,2,3)), 2) / (n ** 2))
    return (first_term - second_term).mean()

def validate(model, loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            img = batch['image'].to(DEVICE)
            depth = batch['depth'].to(DEVICE)

            # Resize GT
            depth = F.interpolate(depth, size=(512, 512), mode='bilinear', align_corners=False)
            depth = torch.log(depth.clamp(min=1e-3))

            pred = model(img)

            # Align Pred
            if pred.shape[-1] != 512:
                pred = F.interpolate(pred, size=(512,512), mode='bilinear', align_corners=False)

            loss = si_loss(pred, depth)
            total_loss += loss.item()
    return total_loss / len(loader)

best_val_loss = float('inf')

for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0

    for i, batch in enumerate(train_loader):
        img = batch['image'].to(DEVICE)
        depth = batch['depth'].to(DEVICE)

        # Resize GT to 512
        depth = F.interpolate(depth, size=(512, 512), mode='bilinear', align_corners=False)
        # Log Depth for SI Loss
        depth = torch.log(depth.clamp(min=1e-3))

        optimizer.zero_grad()
        pred = model(img)

        if pred.shape[-1] != 512:
            pred = F.interpolate(pred, size=(512,512), mode='bilinear', align_corners=False)

        loss = si_loss(pred, depth)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()

        if i % 50 == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] Step [{i}/{len(train_loader)}] Train Loss: {loss.item():.4f}")

    avg_train_loss = running_loss/len(train_loader)

    # Validation step
    val_loss = validate(model, val_loader)
    print(f"Epoch {epoch+1} Finished. Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # SAVE STRATEGY: Minimal Storage
    # 1. Always save 'latest' (Overwrites previous epoch)
    latest_path = f"{SAVE_DIR}/hybrid_latest.pth"
    torch.save(model.state_dict(), latest_path)

    # 2. Save 'best' only if it beats the record
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_path = f"{SAVE_DIR}/hybrid_best.pth"
        torch.save(model.state_dict(), best_path)
        print(f"  >>> New Best Model Saved! (Val Loss: {best_val_loss:.4f})")

print("Phase 3 Complete.")