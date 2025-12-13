import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob

class UniversalDepthDataset(Dataset):
    def __init__(self, dataset_name, split='train', transform=None):
        """
        Args:
            dataset_name: 'nyu' or 'eth3d'
            split: 'train', 'val', or 'test'
        """
        self.dataset_name = dataset_name.lower()
        self.split = split
        self.transform = transform
        self.samples = []

        # --- PATH CONFIGURATION ---
        if self.dataset_name == 'nyu':
            self.root_dir = '/home/junming/nobackup_junming/nyu_depth_v2/extracted_data'
            self._load_nyu()
        elif self.dataset_name == 'eth3d':
            self.root_dir = '/home/junming/nobackup_junming/eth3d-dataset'
            self._load_eth3d()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        print(f"[{dataset_name.upper()}] Split: {split} | Samples: {len(self.samples)}")

    def _load_nyu(self):
        rgb_dir = os.path.join(self.root_dir, 'rgb')
        depth_dir = os.path.join(self.root_dir, 'depth')
        all_rgb = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")))

        # Standard NYU Split
        if self.split == 'train':
            files = all_rgb[:1024]
        elif self.split == 'val':
            files = all_rgb[1024:1248]
        elif self.split == 'test':
            files = all_rgb[1248:]
        else:
            files = []

        for rgb_path in files:
            file_id = os.path.splitext(os.path.basename(rgb_path))[0]
            depth_path = os.path.join(depth_dir, f"{file_id}.npy")
            self.samples.append((rgb_path, depth_path))

    def _load_eth3d(self):
        # ETH3D: Only use the 'train' folder (because 'val' has no GT)
        # We perform an 80/10/10 split manually on the valid data.

        base_dir = os.path.join(self.root_dir, 'train')

        # 1. Robust Image Folder Detection
        img_dir = None
        candidates = ['undistorted', 'distored', 'images', 'raw']
        for cand in candidates:
            check_path = os.path.join(base_dir, cand)
            if os.path.exists(check_path):
                img_dir = check_path
                break

        if not img_dir:
            print(f"Error: No image folder found in {base_dir}")
            return

        gt_dir = os.path.join(base_dir, 'ground_truth')

        # 2. Get All Valid Pairs First
        all_pairs = []
        all_images = sorted(glob.glob(os.path.join(img_dir, "**", "*.[jJ][pP][gG]"), recursive=True))

        for rgb_path in all_images:
            rel_path = os.path.relpath(rgb_path, img_dir)
            file_stem = os.path.splitext(rel_path)[0]

            # Find matching GT
            for ext in ['.pfm', '.png', '.npy']:
                candidate = os.path.join(gt_dir, f"{file_stem}{ext}")
                if os.path.exists(candidate):
                    all_pairs.append((rgb_path, candidate))
                    break

        # 3. Perform the Split (80% Train, 10% Val, 10% Test)
        n_total = len(all_pairs)
        idx_train = int(0.8 * n_total)
        idx_val = int(0.9 * n_total)

        if self.split == 'train':
            self.samples = all_pairs[:idx_train]
        elif self.split == 'val':
            self.samples = all_pairs[idx_train:idx_val]
        elif self.split == 'test':
            self.samples = all_pairs[idx_val:] # Last 10% is our Test Set

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, depth_path = self.samples[idx]

        image = Image.open(rgb_path).convert('RGB')

        if depth_path.endswith('.npy'):
            depth = np.load(depth_path)
        elif depth_path.endswith('.png'):
            depth = np.array(Image.open(depth_path)).astype(np.float32)
            # depth = depth / 1000.0 # Uncomment if needed
        elif depth_path.endswith('.pfm'):
            depth = self._read_pfm(depth_path)
        else:
            depth = np.zeros((image.size[1], image.size[0]), dtype=np.float32)

        depth = torch.from_numpy(depth).float()
        if depth.ndim == 2: depth = depth.unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'depth': depth}

    def _read_pfm(self, path):
        with open(path, 'rb') as file:
            header = file.readline().decode('utf-8').rstrip()
            if header == 'PF': color = True
            elif header == 'Pf': color = False
            else: raise Exception('Not a PFM file.')
            dim_match = file.readline().decode('utf-8').rstrip()
            width, height = map(int, dim_match.split())
            scale = float(file.readline().decode('utf-8').rstrip())
            endian = '<' if scale < 0 else '>'
            data = np.fromfile(file, endian + 'f')
            shape = (height, width, 3) if color else (height, width)
            return np.flipud(np.reshape(data, shape))