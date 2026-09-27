"""
Per-volume statistics for fastMRI HDF5 directories, cached as <data_dir>/volume_stats.json:
    sigma_raw : per-component k-space noise std, median over slices of sqrt(mean |k_edge|^2 / 2) on the top/bottom
                EDGE_ROWS rows of the (image-cropped) k-space, zero-padded columns excluded
    p95_vol   : 95th percentile of the zero-filled |k| over ALL slices of the volume, R=`accel` random mask with a
                fixed seed per slice (the volume-wise fastmri_magnitude scale)
Usage:  python tools/volume_stats.py data/singlecoil_train [--accel 4 --center-fraction 0.08]
"""
import argparse, json, os, sys
from pathlib import Path

import h5py, torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import prepare_fastmri_kspace          # noqa: E402
from train_utils import FastMRIMaskGenerator        # noqa: E402

EDGE_ROWS = 20
STATS_FILENAME = "volume_stats.json"
STATS_SEED = 42


def edge_noise_sigma(kspace):
    """kspace [N, H, W] complex -> median over slices of the edge-row per-component noise std."""
    H, W = kspace.shape[-2:]
    rows = torch.zeros(H, W, dtype=torch.bool); rows[:EDGE_ROWS] = True; rows[-EDGE_ROWS:] = True
    region = rows & (kspace.abs() > 0).all(0).any(0)[None, :]
    edge = kspace[:, region]
    return torch.sqrt((edge.abs() ** 2).mean(1) / 2).median().item()


def volume_p95(kspace, file_index, accel=4, center_fraction=0.08, stride=7):
    """p95 of the zero-filled |k| over the whole volume (deterministic per-slice masks, fixed seed)."""
    gen = FastMRIMaskGenerator([accel], center_fractions=[center_fraction], mask_type="random")
    zf = torch.cat([gen.apply(kspace[s:s + 1, None], accel, seed=(STATS_SEED, file_index, s, accel))[0] for s in range(kspace.shape[0])])
    return torch.quantile(zf.abs().reshape(-1)[::stride], 0.95).item()


def compute_volume_stats(data_dir, image_size=(320, 320), accel=4, center_fraction=0.08, kspace_key="kspace", verbose=True):
    data_dir = Path(data_dir)
    stats = {}
    files = sorted(data_dir.glob("*.h5"))
    for file_index, path in enumerate(tqdm(files, desc=f"volume stats {data_dir.name}", disable=not verbose)):
        with h5py.File(path, "r") as f:
            ks = torch.stack([prepare_fastmri_kspace(torch.as_tensor(f[kspace_key][s], dtype=torch.complex64), image_size)
                              for s in range(f[kspace_key].shape[0])])
        stats[path.name] = {"sigma_raw": edge_noise_sigma(ks), "p95_vol": volume_p95(ks, file_index, accel, center_fraction),
                            "n_slices": int(ks.shape[0])}
    return stats


def load_or_compute_volume_stats(data_dir, image_size=(320, 320), accel=4, center_fraction=0.08, kspace_key="kspace",
                                 path=None, verbose=True):
    """Load <data_dir>/volume_stats.json (or `path`); compute and save it when missing or when files are absent from it."""
    data_dir = Path(data_dir)
    path = Path(path) if path else data_dir / STATS_FILENAME
    stats = json.load(open(path)) if path.is_file() else {}
    missing = [p.name for p in sorted(data_dir.glob("*.h5")) if p.name not in stats]
    if missing:
        if verbose:
            print(f"[volume_stats] computing statistics for {len(missing)} volume(s) in {data_dir} -> {path}")
        stats.update(compute_volume_stats(data_dir, image_size, accel, center_fraction, kspace_key, verbose))
        path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(stats, open(path, "w"), indent=1)
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir"); ap.add_argument("--accel", type=int, default=4); ap.add_argument("--center-fraction", type=float, default=0.08)
    ap.add_argument("--image-size", type=int, nargs=2, default=(320, 320)); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    s = load_or_compute_volume_stats(a.data_dir, tuple(a.image_size), a.accel, a.center_fraction, path=a.out)
    print(f"{len(s)} volumes; sigma_raw median {sorted(v['sigma_raw'] for v in s.values())[len(s)//2]:.3e}; p95_vol median {sorted(v['p95_vol'] for v in s.values())[len(s)//2]:.3e}")
