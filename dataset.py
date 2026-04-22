import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import torch.nn.functional as F

SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.npy'}

try:
    import nibabel as nib
    SUPPORTED_EXTENSIONS.update({'.nii', '.gz'})
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


def load_image(path, N):
    ext = os.path.splitext(path)[1].lower()

    if ext == '.npy':
        img = np.load(path).astype(np.float32)
        if img.ndim == 3:
            img = img[:, :, img.shape[2] // 2]
    elif ext in ('.nii', '.gz') and HAS_NIBABEL:
        vol = nib.load(path).get_fdata().astype(np.float32)
        if vol.ndim == 3:
            img = vol[:, :, vol.shape[2] // 2]
        else:
            img = vol
    else:
        img = np.array(ImageOps.grayscale(Image.open(path))).astype(np.float32)

    # Normalize to [0, 1]
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)

    # Resize to N x N if needed
    if img.shape[0] != N or img.shape[1] != N:
        t = torch.tensor(img).unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(N, N), mode='bilinear', align_corners=False)
        img = t.squeeze().numpy()

    return img


def load_mask(path, N):
    mask = np.array(ImageOps.grayscale(Image.open(path))).astype(np.float32)
    mask = np.fft.ifftshift(mask) / np.max(np.abs(mask))

    if mask.shape[0] != N or mask.shape[1] != N:
        t = torch.tensor(mask).unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(N, N), mode='nearest')
        mask = t.squeeze().numpy()

    return torch.tensor(mask, dtype=torch.float)


class MRIDataset(Dataset):
    """
    Loads fully sampled MRI images and returns ground-truth tensors.
    Undersampling is applied in the training loop so all samples in a
    batch share the same mask.

    Args:
        data_dir (str):              Directory of fully sampled images
        N (int):                     Image size — images are resized to N x N
        split (str):                 'train' or 'val'
        val_fraction (float):        Fraction of data held out for validation
        seed (int):                  Reproducibility seed for the split
    """

    def __init__(self, data_dir, N=320, split='train', val_fraction=0.1, seed=42):
        self.N = N

        all_paths = []
        for fname in sorted(os.listdir(data_dir)):
            ext = os.path.splitext(fname)[1].lower()
            full = os.path.join(data_dir, fname)
            if ext in SUPPORTED_EXTENSIONS or fname.endswith('.nii.gz'):
                all_paths.append(full)

        if not all_paths:
            raise ValueError(f"No supported images found in {data_dir}")

        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(all_paths))
        n_val = max(1, int(len(all_paths) * val_fraction))

        if split == 'val':
            self.paths = [all_paths[i] for i in indices[:n_val]]
        else:
            self.paths = [all_paths[i] for i in indices[n_val:]]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = load_image(self.paths[idx], self.N)
        # [1, N, N] float in [0, 1]
        return torch.tensor(img, dtype=torch.float).unsqueeze(0)
