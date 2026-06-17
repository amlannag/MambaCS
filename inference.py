#!/usr/bin/env python3
"""
Minimal inference helpers for DcTNN experiments.

Given an experiment directory containing a saved checkpoint and config, this
module can:
1. load the trained model
2. run reconstructions on train/val samples
3. save simple side-by-side output images
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from DcTNN.dc import ifft_2d
from dataset import H5MRIDataset
from train_utils import build_model_from_config_dict, generate_column_mask, simulate_undersampling


_DATA_KEYS = {
    "data_dir",
    "val_data_dir",
    "image_size",
    "num_channels",
    "acceleration_factors",
    "val_fraction",
    "seed",
    "kspace_key",
}

_MODEL_KEYS = {
    "encoders",
    "patch_size",
    "nhead_patch",
    "nhead_axial",
    "layer_no",
    "num_encoder_layers",
    "learning",
    "lambda_schedule",
    "pos_emb_type",
    "attn_type",
    "rope_theta",
    "rope_mixed_rotate",
}


def normalise_cfg(cfg_dict):
    if "data" in cfg_dict and "model" in cfg_dict:
        return cfg_dict
    return {
        "data": {k: cfg_dict[k] for k in _DATA_KEYS if k in cfg_dict},
        "model": {k: cfg_dict[k] for k in _MODEL_KEYS if k in cfg_dict},
    }


def resolve_data_dir(data_cfg, split):
    if split == "val" and data_cfg.get("val_data_dir"):
        return data_cfg["val_data_dir"]
    return data_cfg["data_dir"]


def _magnitude_image(tensor):
    array = tensor.detach().cpu().numpy()
    if np.iscomplexobj(array):
        return np.abs(array[0, 0])
    if array.shape[1] == 1:
        return array[0, 0]
    return np.sqrt(array[0, 0] ** 2 + array[0, 1] ** 2)


def _denormalize_image(img_norm, mn, scale):
    """Invert min-max normalisation. Handles complex (k-space) and real (image) tensors."""
    if img_norm.is_complex():
        return torch.complex(img_norm.real * scale + mn, img_norm.imag)
    return img_norm * scale + mn


def _save_reconstruction_figure(output_path, gt_img, zf_img, recon_img, title):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, image, label in zip(
        axes,
        [gt_img, zf_img, recon_img],
        ["Ground Truth", "Zero Filled", "Reconstruction"],
    ):
        ax.imshow(image, cmap="gray")
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_experiment_model(exp_dir, checkpoint_name=None, device=None):
    """
    Load a trained experiment.

    Expected files in exp_dir:
    - config.json
    - best_model.pth or latest.pth
    """
    exp_dir = os.path.abspath(exp_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = os.path.join(exp_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing config.json in {exp_dir}")

    with open(config_path) as f:
        cfg_dict = normalise_cfg(json.load(f))

    if checkpoint_name is None:
        for candidate in ("best_model.pth", "latest.pth"):
            checkpoint_path = os.path.join(exp_dir, candidate)
            if os.path.isfile(checkpoint_path):
                checkpoint_name = candidate
                break
        else:
            raise FileNotFoundError(f"No checkpoint found in {exp_dir}")

    checkpoint_path = os.path.join(exp_dir, checkpoint_name)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    model = build_model_from_config_dict(cfg_dict).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return {
        "model": model,
        "config": cfg_dict,
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "device": device,
    }


@torch.no_grad()
def reconstruct_dataset_samples(
    exp_dir,
    split="val",
    num_images=5,
    accel=None,
    output_dir=None,
    checkpoint_name=None,
):
    """
    Load an experiment, reconstruct a few dataset samples, and save PNG outputs.

    Returns a list of dictionaries with sample metadata and saved file paths.
    """
    loaded = load_experiment_model(exp_dir, checkpoint_name=checkpoint_name)
    model = loaded["model"]
    cfg_dict = loaded["config"]
    device = loaded["device"]

    data_cfg = cfg_dict["data"]
    model_cfg = cfg_dict["model"]
    image_size = data_cfg["image_size"]
    if isinstance(image_size, (tuple, list)):
        img_h, img_w = image_size
    else:
        img_h = img_w = image_size
    learning = model_cfg.get("learning", "k_space")
    if accel is None:
        accel = data_cfg.get("acceleration_factors", [8])[0]

    dataset = H5MRIDataset(
        resolve_data_dir(data_cfg, split),
        image_size=image_size,
        kspace_key=data_cfg.get("kspace_key", "kspace"),
    )

    output_dir = output_dir or os.path.join(
        os.path.abspath(exp_dir),
        f"recon_{split}_R{accel}",
    )
    os.makedirs(output_dir, exist_ok=True)

    num_images = min(num_images, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, num_images, dtype=int)
    mask = generate_column_mask((img_h, img_w), accel, device)

    results = []
    for sample_idx in indices:
        kspace_full = dataset[int(sample_idx)].unsqueeze(0).to(device)
        model_input, kspace_norm, _, stats = simulate_undersampling(kspace_full, mask, learning)
        recon_norm = model(model_input, kspace_norm, mask)

        mn    = stats["min"]
        scale = stats["scale"]
        gt_img = ifft_2d(kspace_full)

        if learning == "k_space":
            zf_img    = _denormalize_image(ifft_2d(kspace_norm), mn, scale)
            recon_img = _denormalize_image(ifft_2d(recon_norm), mn, scale)
        else:
            zf_img    = _denormalize_image(model_input, mn, scale)
            recon_img = _denormalize_image(recon_norm, mn, scale)

        gt_np = _magnitude_image(gt_img)
        zf_np = _magnitude_image(zf_img)
        recon_np = _magnitude_image(recon_img)
        mse = float(np.mean((recon_np - gt_np) ** 2))

        filename = f"sample_{int(sample_idx):05d}.png"
        output_path = os.path.join(output_dir, filename)
        _save_reconstruction_figure(
            output_path,
            gt_np,
            zf_np,
            recon_np,
            title=f"{os.path.basename(os.path.abspath(exp_dir))} | {split} | R={accel} | idx={int(sample_idx)}",
        )

        results.append(
            {
                "index": int(sample_idx),
                "output_path": output_path,
                "mse": mse,
            }
        )

    return results


def run_inference(exp_dir, num_images=5, accel=4, split="val"):
    """
    Backward-compatible wrapper used by train.py.
    """
    results = reconstruct_dataset_samples(
        exp_dir=exp_dir,
        split=split,
        num_images=num_images,
        accel=accel,
    )

    print(f"Saved {len(results)} reconstruction images:")
    for result in results:
        print(f"  idx={result['index']}  mse={result['mse']:.6f}  {result['output_path']}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run simple MRI reconstructions from a saved experiment."
    )
    parser.add_argument("--exp_dir", required=True, help="Experiment directory")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--num_images", type=int, default=5)
    parser.add_argument("--accel", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpoint_name", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    results = reconstruct_dataset_samples(
        exp_dir=args.exp_dir,
        split=args.split,
        num_images=args.num_images,
        accel=args.accel,
        output_dir=args.output_dir,
        checkpoint_name=args.checkpoint_name,
    )
    print(f"Saved {len(results)} reconstruction images.")


if __name__ == "__main__":
    main()
