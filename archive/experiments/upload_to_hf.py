from huggingface_hub import HfApi

# 1. Initialize API
api = HfApi()

# 2. Define your Repo ID (User/RepoName)
repo_id = "Junming111/task3"
file_path = "checkpoints_phase3_nyu/hybrid_best.pth"
target_name = "hybrid_best.pth"

print(f"Checking repository {repo_id}...")

# 3. Create the repo if it doesn't exist
try:
    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
    print("Repository confirmed.")
except Exception as e:
    print(f"Note on Repo Creation: {e}")

# 4. Upload the file
print(f"Starting upload of {file_path} (1.2GB). This may take a few minutes...")
try:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=target_name,
        repo_id=repo_id,
        repo_type="model"
    )
    print("\nSUCCESS! Your model is live at:")
    print(f"https://huggingface.co/{repo_id}/blob/main/{target_name}")
except Exception as e:
    print(f"\nUpload failed: {e}")