import torch
import sys
import os

# 1. Load the File
weight_path = '/home/junming/private/unet-depth-prediction/checkpoints/dinov3_vitl16.pth'
print(f"Loading {weight_path}...")
state_dict = torch.load(weight_path, map_location='cpu')

# Handle nesting
if 'state_dict' in state_dict:
    print("Found 'state_dict' key.")
    state_dict = state_dict['state_dict']
elif 'teacher' in state_dict:
    print("Found 'teacher' key.")
    state_dict = state_dict['teacher']

# Print first 5 keys from FILE
print("\n--- KEYS IN FILE (The Source) ---")
file_keys = list(state_dict.keys())
for k in file_keys[:5]:
    print(f"  {k}")

# 2. Load the Model Structure
sys.path.append('/home/junming/private/dinov3')
model = torch.hub.load('/home/junming/private/dinov3', 'dinov3_vitl16', source='local', pretrained=False)

# Print first 5 keys from MODEL
print("\n--- KEYS IN MODEL (The Destination) ---")
model_keys = list(model.state_dict().keys())
for k in model_keys[:5]:
    print(f"  {k}")