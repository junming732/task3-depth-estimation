from huggingface_hub import HfApi

# 1. Initialize API
api = HfApi()

# 2. Configuration
repo_id = "Junming111/nyu-depth-custom"
file_path = "/home/junming/nobackup_junming/nyu_depth_v2/nyu_data.zip"
target_name = "/home/junming/nobackup_junming/nyu_depth_v2/nyu_data.zip"

print(f"--- Preparing to upload dataset to {repo_id} ---")

# 3. Create the Dataset Repository
try:
    print("Creating repository...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset", # <--- Important!
        exist_ok=True
    )
    print("Repository confirmed.")
except Exception as e:
    print(f"Repo creation warning (might already exist): {e}")

# 4. Upload the Zip File
print(f"Starting upload of {file_path}... (Do not close terminal)")
try:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=target_name,
        repo_id=repo_id,
        repo_type="dataset" # <--- Important!
    )
    print("\nSUCCESS! Dataset uploaded.")
    print(f"Link: https://huggingface.co/datasets/{repo_id}/blob/main/{target_name}")
except Exception as e:
    print(f"\nUpload failed: {e}")