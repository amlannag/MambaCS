import os
import time

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import get_worker_info

from progress import phase, progress_iter


def centered_ifft2(kspace: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(kspace, dim=(-2, -1)), norm='ortho'),
        dim=(-2, -1),
    )


def centered_fft2(image: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(image, dim=(-2, -1)), norm='ortho'),
        dim=(-2, -1),
    )


def center_crop_complex(arr: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    h, w = arr.shape[-2:]
    if not (0 < out_h <= h and 0 < out_w <= w):
        raise ValueError(f"Cannot crop shape {(h, w)} to {(out_h, out_w)}")
    h0 = (h - out_h) // 2
    w0 = (w - out_w) // 2
    return arr[..., h0:h0 + out_h, w0:w0 + out_w]


def prepare_fastmri_kspace(kspace: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    image = centered_ifft2(kspace)
    image = center_crop_complex(image, *image_size)
    return centered_fft2(image)


class H5MRIDataset(Dataset):
    """
    Loads k-space slices from .h5 MRI files (fastMRI format).
    Each file contains kspace of shape (num_slices, H, W) complex64.
    Returns one slice after centered IFFT -> image-domain center crop -> centered FFT,
    as [1, crop_H, crop_W] complex64.

    Args:
        data_dir (str):              Directory containing .h5 files
        image_size (tuple[int,int]): Output image-domain crop shape (crop_H, crop_W)
        kspace_key (str):            HDF5 dataset key for raw k-space (default: 'kspace')
    """

    def __init__(self, data_dir, image_size=(320, 320), kspace_key='kspace', max_files=None,
                 return_metadata=False):
        self.image_size = image_size
        self.kspace_key = kspace_key
        self.return_metadata = return_metadata
        self._file_handles = {}

        h5_files = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith('.h5')
        )
        if not h5_files:
            raise ValueError(f"No .h5 files found in {data_dir}")
        if max_files is not None:
            h5_files = h5_files[:max_files]

        self.h5_files = h5_files
        phase(f"Indexing {len(h5_files)} .h5 files in {data_dir}")
        t_index = time.time()
        self.index = []
        for fpath in progress_iter(h5_files, desc="  indexing", unit="file"):
            with h5py.File(fpath, 'r') as f:
                num_slices = f[kspace_key].shape[0]
            self.index.extend((fpath, s) for s in range(num_slices))
        phase(
            f"Indexed {len(self.index)} slices from {len(h5_files)} files "
            f"in {time.time() - t_index:.1f}s"
        )

    def _get_file_handle(self, fpath):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else None
        handle_key = (worker_id, fpath)
        if handle_key not in self._file_handles:
            self._file_handles[handle_key] = h5py.File(fpath, 'r')
        return self._file_handles[handle_key]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fpath, s = self.index[idx]
        f = self._get_file_handle(fpath)
        kspace = torch.tensor(f[self.kspace_key][s], dtype=torch.complex64)
        kspace = prepare_fastmri_kspace(kspace, self.image_size).unsqueeze(0)
        if not self.return_metadata:
            return kspace
        return {
            "kspace": kspace,
            "fname": os.path.basename(fpath),
            "slice_num": s,
            "max_value": float(f.attrs.get("max", float("nan"))),
        }

    def __del__(self):
        for handle in self._file_handles.values():
            try:
                handle.close()
            except Exception:
                pass


class OASISDataset(Dataset):
    """
    Loads grayscale PNG/JPEG brain slices (OASIS format) and returns centered
    k-space so the downstream pipeline is identical to the fastMRI path.

    Pipeline per sample:
      image [0,1] float32  →  fftshift(fft2(ifftshift(img), norm='ortho'))
                           →  [1, H, W] complex64  (centered k-space)

    The FFT convention matches fft_2d/ifft_2d in DcTNN/dc.py, so
    FastMRIMaskGenerator, simulate_undersampling, and all DC layers need
    no changes.
    """

    def __init__(self, data_dir, image_size=(256, 256), max_files=None):
        self.image_size = image_size

        _EXTS = {'.png', '.jpg', '.jpeg'}
        self.image_files = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if os.path.splitext(f)[1].lower() in _EXTS
        )
        if not self.image_files:
            raise ValueError(f"No PNG/JPEG files found in {data_dir}")
        if max_files is not None:
            self.image_files = self.image_files[:max_files]
        phase(f"Using {len(self.image_files)} images from {data_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert('L')
        img = img.resize((self.image_size[1], self.image_size[0]), Image.LANCZOS)
        img_t = torch.tensor(np.array(img, dtype='float32') / 255.0)   # [H, W] in [0, 1]
        # Convert to centered k-space — same convention as fft_2d in DcTNN/dc.py
        kspace = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(img_t), norm='ortho')
        )
        return kspace.unsqueeze(0).to(torch.complex64)                  # [1, H, W]


class VolumeBatchSampler(torch.utils.data.Sampler):
    """
    Batch sampler that yields all slices of `volumes_per_batch` volumes per batch, so that volume-wise
    operations (e.g. the PCA encoder) see every slice of a volume together. Works with datasets exposing
    an `index` list of (file_path, slice) pairs (H5MRIDataset). Batch sizes vary with the volumes' slice counts.
    """
    def __init__(self, dataset, volumes_per_batch=3, shuffle=True, seed=0, drop_last=False):
        if not hasattr(dataset, "index"):
            raise TypeError("VolumeBatchSampler needs a dataset with an `index` of (file, slice) pairs")
        self.groups = {}
        for i, (fpath, _) in enumerate(dataset.index):
            self.groups.setdefault(fpath, []).append(i)
        self.volumes = list(self.groups)
        self.volumes_per_batch = int(volumes_per_batch)
        self.shuffle, self.seed, self.drop_last = shuffle, seed, drop_last
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        order = list(range(len(self.volumes)))
        if self.shuffle:
            rng = torch.Generator().manual_seed(self.seed + self.epoch)
            order = torch.randperm(len(order), generator=rng).tolist()
            self.epoch += 1
        for start in range(0, len(order), self.volumes_per_batch):
            chunk = order[start:start + self.volumes_per_batch]
            if self.drop_last and len(chunk) < self.volumes_per_batch:
                break
            yield [i for v in chunk for i in self.groups[self.volumes[v]]]

    def __len__(self):
        n = len(self.volumes)
        return n // self.volumes_per_batch if self.drop_last else -(-n // self.volumes_per_batch)


def volume_ids_from_fnames(fnames, device=None):
    """Map a batch's file names to consecutive integer volume ids (same name -> same id)."""
    lookup = {}
    ids = [lookup.setdefault(name, len(lookup)) for name in fnames]
    return torch.tensor(ids, dtype=torch.long, device=device)
