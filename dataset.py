import os
import h5py
import torch
from torch.utils.data import Dataset


def center_crop_kspace(kspace, N):
    """Center-crop a (H, W) complex k-space array to (H, N) — full height, center N columns."""
    H, W = kspace.shape
    w0 = (W - N) // 2
    return kspace[:, w0:w0 + N]


class H5MRIDataset(Dataset):
    """
    Loads k-space slices from .h5 MRI files (fastMRI format).
    Each file contains kspace of shape (num_slices, H, W) complex64.
    Returns one center-cropped slice as [1, H, N] complex64 — full height, center N columns.

    Args:
        data_dir (str):       Directory containing .h5 files
        N (int):              Output width — k-space is center-cropped to H×N
        kspace_key (str):     HDF5 dataset key for raw k-space (default: 'kspace')
    """

    def __init__(self, data_dir, N=320, kspace_key='kspace'):
        self.N = N
        self.kspace_key = kspace_key

        h5_files = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith('.h5')
        )
        if not h5_files:
            raise ValueError(f"No .h5 files found in {data_dir}")

        self.index = []
        for fpath in h5_files:
            with h5py.File(fpath, 'r') as f:
                num_slices = f[kspace_key].shape[0]
            self.index.extend((fpath, s) for s in range(num_slices))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fpath, s = self.index[idx]
        with h5py.File(fpath, 'r') as f:
            kspace = f[self.kspace_key][s]            # (H, W) complex64
        kspace = center_crop_kspace(kspace, self.N)   # (H, N) complex64
        return torch.tensor(kspace, dtype=torch.complex64).unsqueeze(0)  # [1, H, N]
