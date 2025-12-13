import torch
import sys
import os

# 1. Setup Paths
sys.path.append('/home/junming/private/dinov3')
dinov3_repo = '/home/junming/private/dinov3'

print("--- DIAGNOSTICS START ---")

# 2. Load the Model (The Destination)
print("1. Loading Model Structure...")
try:
    model = torch.hub.load(dinov3_repo, 'dinov3_vitl16', source='local', pretrained=False)
    model_keys = list(model.state_dict().keys())
    print(f"   Model Key [0]: '{model_keys[0]}'")
    print(f"   Model Key [5]: '{model_keys[5]}'")
except Exception as e:
    print(f"   Error loading model: {e}")

# 3. Load the File (The Source)
print("\n2. Loading Checkpoint File...")
ckpt_path = '/home/junming/private/unet-depth-prediction/checkpoints/dinov3_vitl16.pth'
if not os.path.exists(ckpt_path):
    print(f"   ERROR: File not found at {ckpt_path}")
else:
    state_dict = torch.load(ckpt_path, map_location='cpu')

    # Unwrap
    if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
    if 'teacher' in state_dict:    state_dict = state_dict['teacher']

    file_keys = list(state_dict.keys())
    print(f"   File Key [0]: '{file_keys[0]}'")
    print(f"   File Key [5]: '{file_keys[5]}'")

print("\n--- DIAGNOSTICS END ---")