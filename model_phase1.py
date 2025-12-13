import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Point to your local DINOv3 clone
sys.path.append('/home/junming/private/dinov3')

class DINOv3LinearProbe(nn.Module):
    def __init__(self, output_size=(224, 224)):
        super().__init__()
        self.output_size = output_size

        print("Loading DINOv3 Backbone (Local Weights)...")

        weight_path = '/home/junming/private/unet-depth-prediction/checkpoints/dinov3_vitl16.pth'

        # 1. Initialize Model
        dinov3_repo = '/home/junming/private/dinov3'
        self.backbone = torch.hub.load(dinov3_repo, 'dinov3_vitl16', source='local', pretrained=False)

        # 2. Load File
        state_dict = torch.load(weight_path, map_location='cpu')

        if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
        if 'teacher' in state_dict:    state_dict = state_dict['teacher']

        # --- THE FIX: Translator + Shape Corrector ---
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k

            # 1. Remove prefixes
            new_k = new_k.replace("module.", "").replace("backbone.", "")

            # 2. Translate Names
            new_k = new_k.replace("embeddings.patch_embeddings", "patch_embed.proj")
            new_k = new_k.replace("embeddings.cls_token", "cls_token")
            new_k = new_k.replace("embeddings.mask_token", "mask_token") # <--- Target Key
            new_k = new_k.replace("embeddings.register_tokens", "storage_tokens")
            new_k = new_k.replace("embeddings.position_embeddings", "pos_embed")
            new_k = new_k.replace("encoder.layers.", "blocks.")
            new_k = new_k.replace("layer_norm1", "norm1")
            new_k = new_k.replace("layer_norm2", "norm2")
            new_k = new_k.replace("layernorm.", "norm.")

            # 3. FIX SHAPE MISMATCH (The Error You Saw)
            # File has [1, 1, 1024], Model wants [1, 1024]
            if "mask_token" in new_k and v.ndim == 3:
                v = v.squeeze(1)

            new_state_dict[new_k] = v

        # 3. Load
        msg = self.backbone.load_state_dict(new_state_dict, strict=False)
        print(f"Weights Loaded. Missing keys: {len(msg.missing_keys)}")

        # Freeze
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.embed_dim = self.backbone.embed_dim

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(self.embed_dim, 512, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 28
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 56
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 112
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 224
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        with torch.no_grad():
            features_dict = self.backbone.forward_features(x)
            patch_tokens = features_dict['x_norm_patchtokens']
            grid_size = int(patch_tokens.shape[1]**0.5)
            feature_map = patch_tokens.permute(0, 2, 1).reshape(B, self.embed_dim, grid_size, grid_size)

        depth = self.decoder(feature_map)

        if depth.shape[-1] != self.output_size[0]:
            depth = F.interpolate(depth, size=self.output_size, mode='bilinear')
        return depth