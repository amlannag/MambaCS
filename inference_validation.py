"""Evaluate trained experiments over every validation slice and save RSS metrics."""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

from dataset import H5MRIDataset, OASISDataset, VolumeBatchSampler, volume_ids_from_fnames
from inference import load_experiment_model
from tools.volume_stats import load_or_compute_volume_stats
from train_utils import (
    FastMRIMaskGenerator,
    is_h5_dataset,
    is_multicoil_dataset,
    resolve_data_dirs,
    simulate_undersampling,
)
from normalizer import reconstruction_to_image_magnitude


def _default_experiments_dir():
    configured = os.environ.get("MAMBACS_EXPERIMENTS_DIR")
    return Path(configured).expanduser() if configured else Path.home() / "Experiments"


def _slice_metric(target, prediction):
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    peak = float(np.max(target))
    mse = float(np.mean(np.square(prediction - target)))
    psnr = float("inf") if mse == 0 else 20.0 * math.log10(max(peak, 1e-12) / math.sqrt(mse))
    data_range = max(peak, 1e-12)
    ssim = float(structural_similarity(target, prediction, data_range=data_range))
    return psnr, ssim, peak


def _rss_image(coil_magnitude):
    if coil_magnitude.ndim != 4:
        raise ValueError(f"Expected [batch, coils, height, width], got {tuple(coil_magnitude.shape)}")
    return coil_magnitude.square().sum(dim=1).sqrt()


def _batch_names(batch, batch_size, first_index):
    if isinstance(batch, dict) and "fname" in batch:
        names = batch["fname"]
        if isinstance(names, str):
            names = [names]
        return [str(name) for name in names]
    return [f"slice_{first_index + offset:06d}" for offset in range(batch_size)]


def _batch_slice_numbers(batch, batch_size, first_index):
    if isinstance(batch, dict) and "slice_num" in batch:
        numbers = batch["slice_num"]
        if torch.is_tensor(numbers):
            return [int(number) for number in numbers.tolist()]
        if isinstance(numbers, (list, tuple)):
            return [int(number) for number in numbers]
        return [int(numbers)]
    return [first_index + offset for offset in range(batch_size)]


def _needs_volume_stats(cfg):
    return (
        getattr(cfg, "norm_scope", "slice") == "volume"
        or "ring_phase_mag" in (cfg.final_loss_type, cfg.intermediate_loss_type)
    )


def _run_experiment(experiment_dir, val_dir_override, device, batch_size, num_workers):
    experiment_dir = experiment_dir.resolve()
    checkpoint_path = experiment_dir / "best_model.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing best checkpoint: {checkpoint_path}")

    loaded = load_experiment_model(str(experiment_dir), device=device)
    model, cfg = loaded["model"], loaded["cfg"]
    if val_dir_override:
        val_dir = str(Path(val_dir_override).expanduser().resolve())
    else:
        _, val_dir = resolve_data_dirs(cfg)
    if not Path(val_dir).is_dir():
        raise FileNotFoundError(f"Validation directory does not exist: {val_dir}")

    if cfg.dataset == "oasis":
        dataset = OASISDataset(val_dir, image_size=tuple(cfg.image_size), max_files=None)
        volume_stats = None
    elif is_h5_dataset(cfg):
        volume_stats = None
        if _needs_volume_stats(cfg):
            if is_multicoil_dataset(cfg):
                raise ValueError("Volume-scoped normalization/statistics are not supported for multicoil MRI.")
            volume_stats = load_or_compute_volume_stats(
                val_dir,
                tuple(cfg.image_size),
                int(cfg.acceleration_factors[0]),
                float(cfg.center_fractions[0]),
                cfg.kspace_key,
                path=getattr(cfg, "volume_stats_path", None),
            )
        dataset = H5MRIDataset(
            val_dir,
            image_size=tuple(cfg.image_size),
            kspace_key=cfg.kspace_key,
            max_files=None,
            return_metadata=True,
            volume_stats=volume_stats,
            num_coils=cfg.num_channels if is_multicoil_dataset(cfg) else None,
        )
    else:
        raise ValueError(f"Unsupported dataset {cfg.dataset!r} in the experiment config.")

    volume_batches = "pca" in cfg.encoders and getattr(cfg, "pca_scope", "volume") == "volume"
    if volume_batches and not is_h5_dataset(cfg):
        raise ValueError("Volume-scoped PCA inference requires an HDF5 validation dataset.")
    if volume_batches:
        loader = DataLoader(
            dataset,
            batch_sampler=VolumeBatchSampler(
                dataset,
                volumes_per_batch=1,
                shuffle=False,
                seed=cfg.seed,
            ),
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
    mask_generator = FastMRIMaskGenerator(
        cfg.acceleration_factors,
        center_fractions=cfg.center_fractions,
        mask_type=cfg.mask_type,
    )
    is_pca = "pca" in cfg.encoders
    per_slice = []
    first_index = 0
    started = time.time()
    model.eval()

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            kspace_full = batch["kspace"] if isinstance(batch, dict) else batch
            kspace_full = kspace_full.to(device, non_blocking=device.type == "cuda")
            current_batch_size = int(kspace_full.shape[0])
            filenames = _batch_names(batch, current_batch_size, first_index)
            slice_numbers = _batch_slice_numbers(batch, current_batch_size, first_index)
            acceleration = int(cfg.acceleration_factors[batch_index % len(cfg.acceleration_factors)])
            kspace_us, mask, num_low_frequencies = mask_generator.apply(
                kspace_full,
                acceleration,
                seed=(cfg.seed, batch_index, acceleration),
            )

            volume_kwargs = {}
            if isinstance(batch, dict) and "sigma_raw" in batch:
                volume_kwargs["sigma_raw"] = torch.as_tensor(
                    batch["sigma_raw"], dtype=torch.float32, device=device,
                )
                if getattr(cfg, "norm_scope", "slice") == "volume":
                    volume_kwargs["volume_scale"] = torch.as_tensor(
                        batch["p95_vol"], dtype=torch.float32, device=device,
                    )

            model_input, dc_input, target, stats = simulate_undersampling(
                kspace_full,
                mask,
                learning=cfg.learning,
                norm=cfg.norm,
                kspace_us=kspace_us,
                kspace_fill=cfg.kspace_fill,
                num_low_frequencies=num_low_frequencies,
                grappa_phase_lines=cfg.grappa_phase_lines,
                grappa_readout_width=cfg.grappa_readout_width,
                grappa_ridge=cfg.grappa_ridge,
                robust_clip=cfg.robust_clip,
                robust_shift=cfg.robust_shift,
                companding_p=cfg.companding_p,
                companding_a=cfg.companding_a,
                companding_centering=cfg.companding_centering,
                norm_quantile=cfg.norm_quantile,
                **volume_kwargs,
            )
            volume_ids = volume_ids_from_fnames(filenames, device=device) if is_pca else None
            reconstruction = model(
                model_input,
                dc_input,
                mask,
                stats=stats,
                volume_id=volume_ids,
            )
            predicted_coil_images = reconstruction_to_image_magnitude(reconstruction, stats)
            predicted_rss = _rss_image(predicted_coil_images).cpu().numpy()
            target_rss = _rss_image(target["image"]).cpu().numpy()

            for offset in range(current_batch_size):
                psnr, ssim, peak = _slice_metric(target_rss[offset], predicted_rss[offset])
                per_slice.append({
                    "filename": filenames[offset],
                    "slice_index": slice_numbers[offset],
                    "acceleration": acceleration,
                    "rss_psnr_db": psnr,
                    "rss_ssim": ssim,
                    "target_rss_peak": peak,
                })
            first_index += current_batch_size
            if (batch_index + 1) % 100 == 0 or batch_index + 1 == len(loader):
                print(
                    f"  {experiment_dir.name}: processed {first_index}/{len(dataset)} slices",
                    flush=True,
                )

    if not per_slice:
        raise ValueError(f"No validation slices found in {val_dir}.")

    def summarize(records):
        return {
            "slice_count": len(records),
            "mean_rss_psnr_db": float(np.mean([record["rss_psnr_db"] for record in records])),
            "mean_rss_ssim": float(np.mean([record["rss_ssim"] for record in records])),
        }

    by_acceleration = {
        str(acceleration): summarize([
            record for record in per_slice if record["acceleration"] == acceleration
        ])
        for acceleration in sorted({record["acceleration"] for record in per_slice})
    }
    overall = summarize(per_slice)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = experiment_dir / "inference_runs"
    run_root.mkdir(parents=True, exist_ok=True)
    run_dir = run_root / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = run_root / f"{timestamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir()

    result = {
        "experiment": experiment_dir.name,
        "experiment_dir": str(experiment_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": loaded["checkpoint"].get("epoch"),
        "dataset": cfg.dataset,
        "validation_dir": str(val_dir),
        "slice_count": overall["slice_count"],
        "mask_type": cfg.mask_type,
        "acceleration_factors": [int(value) for value in cfg.acceleration_factors],
        "center_fractions": (
            [float(value) for value in cfg.center_fractions]
            if cfg.center_fractions is not None else None
        ),
        "seed": int(cfg.seed),
        "metric_definition": (
            "Mean of per-slice PSNR and SSIM on RSS-combined coil magnitude images; "
            "PSNR peak and SSIM data_range are each slice's target RSS maximum."
        ),
        "overall": overall,
        "by_acceleration": by_acceleration,
        "runtime_seconds": round(time.time() - started, 2),
        "per_slice_csv": str(run_dir / "per_slice_metrics.csv"),
    }
    with (run_dir / "per_slice_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_slice[0]))
        writer.writeheader()
        writer.writerows(per_slice)
    with (run_dir / "summary.json").open("w") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)

    print(
        f"{experiment_dir.name}: mean RSS PSNR={overall['mean_rss_psnr_db']:.3f} dB, "
        f"mean RSS SSIM={overall['mean_rss_ssim']:.6f} "
        f"({overall['slice_count']} slices; {time.time() - started:.1f}s)"
    )
    print(f"  Results: {run_dir}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run full-validation-set RSS PSNR/SSIM inference for Bunya experiment folders."
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="Experiment folder names or relative paths beneath --experiments-dir.",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=_default_experiments_dir(),
        help="Bunya Experiments root (default: $HOME/Experiments).",
    )
    parser.add_argument(
        "--val-dir",
        default=None,
        help="Override the validation directory saved in each experiment config.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("--batch-size must be positive and --num-workers must be non-negative.")
    experiments_root = args.experiments_dir.expanduser().resolve()
    if not experiments_root.is_dir():
        parser.error(f"Experiments directory does not exist: {experiments_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Experiments root: {experiments_root}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} (ROCm/HIP={torch.version.hip})")
    summaries = []
    for name in args.experiments:
        experiment_dir = (experiments_root / name).resolve()
        if experiments_root not in experiment_dir.parents:
            parser.error(f"Experiment path must be inside {experiments_root}: {name}")
        if not experiment_dir.is_dir():
            raise FileNotFoundError(f"Experiment folder does not exist: {experiment_dir}")
        summaries.append(_run_experiment(
            experiment_dir,
            args.val_dir,
            device,
            args.batch_size,
            args.num_workers,
        ))

    summary_root = experiments_root / "inference_summaries"
    summary_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = summary_root / f"rss_metrics_{timestamp}.csv"
    suffix = 1
    while summary_path.exists():
        summary_path = summary_root / f"rss_metrics_{timestamp}_{suffix:02d}.csv"
        suffix += 1
    with summary_path.open("w", newline="") as handle:
        fields = [
            "experiment",
            "checkpoint_epoch",
            "dataset",
            "slice_count",
            "mean_rss_psnr_db",
            "mean_rss_ssim",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({
                "experiment": summary["experiment"],
                "checkpoint_epoch": summary["checkpoint_epoch"],
                "dataset": summary["dataset"],
                **summary["overall"],
            })
    print(f"Combined summary: {summary_path}")


if __name__ == "__main__":
    main()
