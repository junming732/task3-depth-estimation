from huggingface_hub import HfApi
import os

# 1. Initialize API
api = HfApi()
repo_id = "Junming111/nyu-depth-custom"

# 2. Define Explicit Paths
# SOURCE: Where the file is on your UPPMAX server (Absolute Path)
local_file_path = "/home/junming/nobackup_junming/nyu_depth_v2/nyu_data.zip"

# DESTINATION: Where it should go on Hugging Face (Clean Root Name)
target_filename = "nyu_data.zip"

# 3. Safety Check
if not os.path.exists(local_file_path):
    print(f"ERROR: Could not find file at: {local_file_path}")
    exit()

# 4. Upload
print(f"Uploading from: {local_file_path}")
print(f"Targeting: {repo_id}/{target_filename} (Root level)")

try:
    api.upload_file(
        path_or_fileobj=local_file_path,
        path_in_repo=target_filename, # <--- This fixes the folder nesting!
        repo_id=repo_id,
        repo_type="dataset"
    )
    print("\nSUCCESS! Dataset fixed.")
    print(f"Clean Link: https://huggingface.co/datasets/{repo_id}/blob/main/{target_filename}")
except Exception as e:
    print(f"Upload failed: {e}")