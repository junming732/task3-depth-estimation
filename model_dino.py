import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Make DINOv3 importable
sys.path.append(os.path.join(os.getcwd(), 'dinov3'))

class DINOv3Depth(nn.Module):
    def __init__(self, output_size=(224, 224)):
        super().__init__()
        self.output_size = output_size

        # --- THE CHANGE: Load DINOv3 ---
        print("Loading REAL DINOv3 Backbone...")
        # Option A: If using local clone (Best for UPPMAX)
        # You might need to check the specific DINOv3 repo structure (e.g., hubconf.py)
        # Often it's: torch.hub.load(local_dir, 'dinov3_vitl14', source='local')
        self.backbone = torch.hub.load('./dinov3', 'dinov3_vitl14', source='local', pretrained=True)

        # Freeze it
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.embed_dim = self.backbone.embed_dim

        # --- Decoder (Same as before) ---
        self.decoder = nn.Sequential(
            nn.Conv2d(self.embed_dim, 512, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 32
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 64
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 128
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=1.75, mode='bilinear', align_corners=False), # 224
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        # ... (Same forward logic as High-Res version) ...
        # DINOv3 API usually matches DINOv2: forward_features(x)['x_norm_patchtokens']
        # Double check if DINOv3 requires 'registers' or specific keys

        with torch.no_grad():
            features_dict = self.backbone.forward_features(x)
            patch_tokens = features_dict['x_norm_patchtokens']
            B, N, C = patch_tokens.shape

            # Calculate grid size dynamically (sq root of patches)
            # e.g. 224->256 patches->16x16
            grid_size = int(N**0.5)
            feature_map = patch_tokens.permute(0, 2, 1).reshape(B, C, grid_size, grid_size)

        depth = self.decoder(feature_map)

        if depth.shape[-1] != 224:
            depth = F.interpolate(depth, size=(224, 224), mode='bilinear')

        return depth