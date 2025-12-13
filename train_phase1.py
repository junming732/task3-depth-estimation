import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import os
import argparse

from dataset_universal import UniversalDepthDataset
from model_phase1 import DINOv3LinearProbe

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, choices=['nyu', 'eth3d'])
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-4)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = f'checkpoints_phase1_{args.dataset}'
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"--- Starting Phase 1 Training on {args.dataset.upper()} ---")

dino_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_ds = UniversalDepthDataset(dataset_name=args.dataset, split='train', transform=dino_transform)
val_ds   = UniversalDepthDataset(dataset_name=args.dataset, split='val', transform=dino_transform)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

model = DINOv3LinearProbe(output_size=(224, 224)).to(DEVICE)
optimizer = optim.AdamW(model.decoder.parameters(), lr=args.lr)

# --- FIXED LOSS FUNCTION ---
def si_loss(pred, target):
    # Scale Invariant Loss (Eigen et al.)
    # Input: (B, 1, H, W)

    # 1. Difference in Log Space
    di = pred - target

    # 2. Number of valid pixels per image
    n = (di.shape[-1] * di.shape[-2]) # H * W

    # 3. Calculate per image to avoid batch explosion
    di2 = torch.pow(di, 2)

    # Sum of squares per image
    first_term = torch.sum(di2, dim=(1,2,3)) / n

    # Square of sums per image
    second_term = 0.5 * (torch.pow(torch.sum(di, dim=(1,2,3)), 2) / (n ** 2))

    # 4. Average over batch
    loss = first_term - second_term
    return loss.mean()

for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0

    for i, batch in enumerate(train_loader):
        img = batch['image'].to(DEVICE)
        depth = batch['depth'].to(DEVICE)

        depth = F.interpolate(depth, size=(224, 224), mode='bilinear', align_corners=False)
        depth = torch.log(depth.clamp(min=1e-3))

        optimizer.zero_grad()
        pred = model(img)
        loss = si_loss(pred, depth)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            img = batch['image'].to(DEVICE)
            depth = batch['depth'].to(DEVICE)
            depth = F.interpolate(depth, size=(224, 224), mode='bilinear', align_corners=False)
            depth = torch.log(depth.clamp(min=1e-3))

            pred = model(img)
            val_loss += si_loss(pred, depth).item()

    avg_val = val_loss / len(val_loader)
    print(f"--- Epoch {epoch+1} Val Loss: {avg_val:.4f} ---")
    torch.save(model.state_dict(), f"{SAVE_DIR}/model_epoch_{epoch+1}.pth")

print("Training Complete.")