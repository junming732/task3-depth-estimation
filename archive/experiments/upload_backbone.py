from huggingface_hub import HfApi
import os

# 1. Initialize API
api = HfApi()

# 2. Configuration
repo_id = "Junming111/task3"  # Same repo where your model is
file_path = "checkpoints/dinov3_vitl16.pth" # Path based on your 'ls -lh'
target_name = "dinov3_vitl16.pth"

print(f"--- Preparing to upload Backbone to {repo_id} ---")

if not os.path.exists(file_path):
    print(f"ERROR: Could not find file at {file_path}")
    print("Please check if the file is in 'checkpoints/' or the current folder.")
    exit()

# 3. Upload the File
print(f"Starting upload of {file_path} (1.2GB).")
print("This will take a few minutes. Do not close the terminal...")

try:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=target_name,
        repo_id=repo_id,
        repo_type="model"
    )
    print("\nSUCCESS! Backbone uploaded.")
    print(f"Link: https://huggingface.co/{repo_id}/blob/main/{target_name}")
except Exception as e:
    print(f"\nUpload failed: {e}")