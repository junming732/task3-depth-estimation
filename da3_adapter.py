import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# 1. Path to DINOv3
sys.path.append('/home/junming/private/dinov3')

class DPTHead(nn.Module):
    def __init__(self, in_channels, features=256):
        super().__init__()
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channel, features, kernel_size=1, bias=False)
            for in_channel in in_channels
        ])

        # --- THE CHECKERBOARD FIX ---
        # Replaced ConvTranspose with Bilinear Upsampling to force smoothness
        self.resize_layers = nn.ModuleList([
            # Layer 4 (x4 Upsample)
            nn.Sequential(
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
                nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False)
            ),
            # Layer 11 (x2 Upsample)
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False)
            ),
            # Layer 17 (Identity)
            nn.Identity(),
            # Layer 23 (Downsample)
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])

        self.fusion = nn.Sequential(
            nn.Conv2d(features * 4, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True)
        )
        self.output_head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(features // 2, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, features):
        out = []

        # 1. Get Target Size from the largest map (Layer 4)
        # This prevents the "Blob" issue by ensuring we target 128x128 resolution, not 32x32
        f0 = features[0]
        x0 = self.projects[0](f0)
        x0 = self.resize_layers[0](x0)
        target_H, target_W = x0.shape[-2:]

        for i, f in enumerate(features):
            x = self.projects[i](f)
            x = self.resize_layers[i](x)

            # Force all maps to align to the largest size
            if x.shape[-2:] != (target_H, target_W):
                x = F.interpolate(x, size=(target_H, target_W), mode='bilinear', align_corners=False)
            out.append(x)

        out = torch.cat(out, dim=1)
        out = self.fusion(out)
        out = self.output_head(out)
        return out

class DINOv3_DA3_Hybrid(nn.Module):
    def __init__(self):
        super().__init__()
        print("Loading DINOv3 Backbone for Phase 3...")

        dinov3_repo = '/home/junming/private/dinov3'
        self.backbone = torch.hub.load(dinov3_repo, 'dinov3_vitl16', source='local', pretrained=False)

        weight_path = '/home/junming/private/unet-depth-prediction/checkpoints/dinov3_vitl16.pth'
        state_dict = torch.load(weight_path, map_location='cpu')

        if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
        if 'teacher' in state_dict:    state_dict = state_dict['teacher']

        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace("module.", "").replace("backbone.", "")
            new_k = new_k.replace("embeddings.patch_embeddings", "patch_embed.proj")
            new_k = new_k.replace("embeddings.cls_token", "cls_token")
            new_k = new_k.replace("embeddings.mask_token", "mask_token")
            new_k = new_k.replace("embeddings.register_tokens", "storage_tokens")
            new_k = new_k.replace("embeddings.position_embeddings", "pos_embed")
            new_k = new_k.replace("encoder.layers.", "blocks.")
            new_k = new_k.replace("layer.", "blocks.")
            new_k = new_k.replace("attention.", "attn.")
            new_k = new_k.replace("intermediate.dense", "mlp.fc1")
            new_k = new_k.replace("output.dense", "mlp.fc2")
            new_k = new_k.replace("layernorm.", "norm.")
            new_k = new_k.replace("layer_norm1", "norm1")
            new_k = new_k.replace("layer_norm2", "norm2")
            if "mask_token" in new_k and v.ndim == 3: v = v.squeeze(1)
            new_state_dict[new_k] = v

        # FUSE QKV
        final_state_dict = {}
        fused_keys = set()
        for i in range(24):
            prefix = f"blocks.{i}.attn"
            q_key, k_key, v_key = f"{prefix}.q_proj.weight", f"{prefix}.k_proj.weight", f"{prefix}.v_proj.weight"
            if q_key in new_state_dict and k_key in new_state_dict and v_key in new_state_dict:
                q, k, v = new_state_dict[q_key], new_state_dict[k_key], new_state_dict[v_key]
                final_state_dict[f"{prefix}.qkv.weight"] = torch.cat([q, k, v], dim=0)
                fused_keys.update([q_key, k_key, v_key])
                q_b, k_b, v_b = f"{prefix}.q_proj.bias", f"{prefix}.k_proj.bias", f"{prefix}.v_proj.bias"
                if q_b in new_state_dict and k_b in new_state_dict and v_b in new_state_dict:
                    qb, kb, vb = new_state_dict[q_b], new_state_dict[k_b], new_state_dict[v_b]
                    final_state_dict[f"{prefix}.qkv.bias"] = torch.cat([qb, kb, vb], dim=0)
                    fused_keys.update([q_b, k_b, v_b])

        for k, v in new_state_dict.items():
            if k not in fused_keys:
                final_state_dict[k] = v

        msg = self.backbone.load_state_dict(final_state_dict, strict=False)
        print(f"Backbone Loaded. Missing keys: {len(msg.missing_keys)}")

        # --- UNFREEZE STRATEGY FOR 30 EPOCHS ---
        # We unfreeze the last 3 blocks (21, 22, 23) to allow better adaptation
        for param in self.backbone.parameters():
            param.requires_grad = False

        for name, param in self.backbone.named_parameters():
            if any(x in name for x in ["blocks.21", "blocks.22", "blocks.23", "norm"]):
                param.requires_grad = True

        print("Model Configuration: Unfrozen Blocks 21, 22, 23 + Norms.")
        self.head = DPTHead(in_channels=[1024, 1024, 1024, 1024])

    def forward(self, x):
        # 1. Get ALL layers
        all_feat = self.backbone.get_intermediate_layers(x, n=24, reshape=True)

        # 2. Pick the "Golden Quartet"
        # Layer 4  = Edges (High Res)
        # Layer 11 = Shapes
        # Layer 17 = Objects
        # Layer 23 = Context
        features = [all_feat[4], all_feat[11], all_feat[17], all_feat[23]]

        # 3. Pass to Head
        depth = self.head(features)
        return depth