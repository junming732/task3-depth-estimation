from huggingface_hub import HfApi

api = HfApi()
repo_id = "Junming111/nyu-depth-custom"

# The accidental long path (based on your logs)
# Note: We point to the 'home' folder to delete everything inside it
folder_to_delete = "home"

print(f"Cleaning up {repo_id}...")

try:
    # This deletes the 'home' folder and all subfolders inside it
    api.delete_folder(
        path_in_repo=folder_to_delete,
        repo_id=repo_id,
        repo_type="dataset"
    )
    print("SUCCESS: Deleted the messy 'home' directory.")
except Exception as e:
    print(f"Error: {e}")
    print("Tip: If 'delete_folder' fails, the path might be slightly different.")
    print("Try going to the website URL to check the exact folder name.")