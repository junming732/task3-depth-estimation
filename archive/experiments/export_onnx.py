import torch
import torch.onnx
import sys
import os

# Ensure we can find the local modules
sys.path.append(os.getcwd())

# Import your model
from model_phase1 import DINOv3LinearProbe

def export():
    print("--- Phase 2: Generating Architecture Visualization ---")

    # 1. Initialize Model
    # (This will trigger your 'Shim' loader, which is fine)
    try:
        model = DINOv3LinearProbe(output_size=(224, 224))
        model.eval()
    except Exception as e:
        print(f"\nERROR: Model failed to load. Are paths correct in model_phase1.py?\nDetails: {e}")
        return

    # 2. Create Dummy Input (1 Image, 3 Channels, 224x224)
    dummy_input = torch.randn(1, 3, 224, 224)

    # 3. Export to ONNX
    output_file = "phase1_arch.onnx"
    print(f"Exporting to {output_file}...")

    torch.onnx.export(
        model,
        dummy_input,
        output_file,
        input_names=['RGB_Input'],
        output_names=['Depth_Output'],
        opset_version=11
    )

    print(f"\nSUCCESS! Saved {output_file}")
    print("Now download this file to your laptop and open it in Netron.app")

if __name__ == "__main__":
    export()