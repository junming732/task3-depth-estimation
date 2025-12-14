import os
import glob

# --- CONFIG ---
# Your exact path to the train folder
ROOT = '/home/junming/nobackup_junming/eth3d-dataset/train'

def inspect():
    print(f"--- INSPECTING: {ROOT} ---")

    # 1. Check Image Folder (We know 'distored' exists from your ls)
    img_dir = os.path.join(ROOT, 'distored')
    if not os.path.exists(img_dir):
        print(f"CRITICAL: Image dir not found at {img_dir}")
        return
    else:
        print(f"Found Image Dir: {img_dir}")

    # 2. Check GT Folder
    gt_dir = os.path.join(ROOT, 'ground_truth')
    if not os.path.exists(gt_dir):
        print(f"CRITICAL: GT dir not found at {gt_dir}")
        return
    else:
        print(f"Found GT Dir:    {gt_dir}")

    # 3. List first 3 Images
    print("\n--- SAMPLE IMAGES ---")
    # Search recursively for jpg/png/JPG
    images = sorted(glob.glob(os.path.join(img_dir, "**", "*.[jJ][pP][gG]"), recursive=True))

    if len(images) == 0:
        print("No images found! Check permissions or file extensions.")
        return

    # Print first few to see structure
    for i in range(min(3, len(images))):
        rel = os.path.relpath(images[i], img_dir)
        print(f"  Img: {rel}")

    # 4. List first 3 Ground Truth files
    print("\n--- SAMPLE DEPTH FILES ---")
    # Search for anything in GT to see the structure
    depths = sorted(glob.glob(os.path.join(gt_dir, "**", "*"), recursive=True))
    # Filter for files only (skip folders)
    depths = [d for d in depths if os.path.isfile(d)]

    for i in range(min(5, len(depths))):
        rel = os.path.relpath(depths[i], gt_dir)
        print(f"  GT:  {rel}")

    # 5. Try to Match the first image
    print("\n--- MATCHING TEST ---")
    test_img = images[0]
    rel_path = os.path.relpath(test_img, img_dir)
    file_stem = os.path.splitext(rel_path)[0]

    print(f"Trying to match: {rel_path}")
    print(f"Looking for stem: {file_stem}")

    extensions = ['.pfm', '.png', '.npy']
    found = False
    for ext in extensions:
        candidate = os.path.join(gt_dir, f"{file_stem}{ext}")
        print(f"  Checking: {candidate}")
        if os.path.exists(candidate):
            print("  -> FOUND!")
            found = True
            break

    if not found:
        print("  -> FAILED to find matching depth file.")
        print("  COMPARE 'Img' and 'GT' paths above. Do the folder names match?")

if __name__ == "__main__":
    inspect()