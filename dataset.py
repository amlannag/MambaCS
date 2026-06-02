import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import torch.nn.functional as F


def load_mask(path, N):
    mask = np.array(ImageOps.grayscale(Image.open(path))).astype(np.float32)
    mask = np.fft.ifftshift(mask) / np.max(np.abs(mask))

    if mask.shape[0] != N or mask.shape[1] != N:
        t = torch.tensor(mask).unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(N, N), mode='nearest')
        mask = t.squeeze().numpy()

    return torch.tensor(mask, dtype=torch.float)


def center_crop_kspace(kspace, N):
    """Center-crop a (H, W) complex k-space array to (N, N)."""
    H, W = kspace.shape
    h0 = (H - N) // 2
    w0 = (W - N) // 2
    return kspace[h0:h0 + N, w0:w0 + N]


class H5MRIDataset(Dataset):
    """
    Loads k-space slices from .h5 MRI files (fastMRI format).
    Each file contains kspace of shape (num_slices, H, W) complex64.
    Returns one center-cropped slice as [2, N, N] float32 (real + imag channels).

    Train/val split is at the file level to prevent data leakage between
    adjacent slices from the same scan.

    Args:
        data_dir (str):       Directory containing .h5 files
        N (int):              Output spatial size — k-space is center-cropped to N×N
        split (str):          'train' or 'val'
        val_fraction (float): Fraction of files held out for validation
        seed (int):           Reproducibility seed for the file-level split
        kspace_key (str):     HDF5 dataset key for raw k-space (default: 'kspace')
    """

    def __init__(self, data_dir, N=320, split='train', val_fraction=0.1, seed=42,
                 kspace_key='kspace'):
        self.N = N
        self.kspace_key = kspace_key

        h5_files = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith('.h5')
        )
        if not h5_files:
            raise ValueError(f"No .h5 files found in {data_dir}")

        rng = np.random.RandomState(seed)
        file_indices = rng.permutation(len(h5_files))
        n_val = max(1, int(len(h5_files) * val_fraction))

        if split == 'val':
            chosen_files = [h5_files[i] for i in file_indices[:n_val]]
        else:
            chosen_files = [h5_files[i] for i in file_indices[n_val:]]

        self.index = []
        for fpath in chosen_files:
            with h5py.File(fpath, 'r') as f:
                num_slices = f[kspace_key].shape[0]
            self.index.extend((fpath, s) for s in range(num_slices))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fpath, s = self.index[idx]
        with h5py.File(fpath, 'r') as f:
            kspace = f[self.kspace_key][s]            # (H, W) complex64
        kspace = center_crop_kspace(kspace, self.N)   # (N, N) complex64
        real = torch.tensor(kspace.real, dtype=torch.float32)
        imag = torch.tensor(kspace.imag, dtype=torch.float32)
        return torch.stack([real, imag], dim=0)       # [2, N, N]
