import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import os
import argparse

from dataset_universal import UniversalDepthDataset
from da3_adapter import DINOv3_DA3_Hybrid

# --- Args ---
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, choices=['nyu', 'eth3d'])
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--lr', type=float, default=5e-5)
args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = f'checkpoints_phase3_{args.dataset}'
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"--- Starting Phase 3 (Hybrid) on {args.dataset.upper()} ---")

# --- CORRECT TRANSFORM (512x512) ---
# 512 is divisible by Patch Size 16. 518 is NOT.
da3_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_loader = DataLoader(
    UniversalDepthDataset(args.dataset, 'train', transform=da3_transform),
    batch_size=args.batch_size, shuffle=True, num_workers=4
)
val_loader = DataLoader(
    UniversalDepthDataset(args.dataset, 'val', transform=da3_transform),
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

for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0

    for i, batch in enumerate(train_loader):
        img = batch['image'].to(DEVICE)
        depth = batch['depth'].to(DEVICE)

        # Resize GT to 512
        depth = F.interpolate(depth, size=(512, 512), mode='bilinear', align_corners=False)
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

        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] Step [{i}/{len(train_loader)}] Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} Finished. Avg Loss: {running_loss/len(train_loader):.4f}")
    torch.save(model.state_dict(), f"{SAVE_DIR}/hybrid_epoch_{epoch+1}.pth")

print("Phase 3 Complete.")