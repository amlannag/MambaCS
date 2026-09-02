import os
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import get_worker_info


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
        self.index = []
        for fpath in h5_files:
            with h5py.File(fpath, 'r') as f:
                num_slices = f[kspace_key].shape[0]
            self.index.extend((fpath, s) for s in range(num_slices))

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
