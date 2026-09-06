"""
Training pipeline for Mamba Compressed Sensing MRI reconstruction.
"""

import argparse
import dataclasses
import gc
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

from dataset import H5MRIDataset, OASISDataset
from config import Config
from progress import phase, progress_iter
from train_config import EXPERIMENTS
from DcTNN.lambda_scheduler import LambdaScheduler
from train_utils import FastMRIMaskGenerator, build_model, resolve_data_dirs, simulate_undersampling
from DcTNN.loss import PerpendicularLoss, build_loss
from normalizer import reconstruction_to_image_magnitude


def build_cfg(exp_idx: int) -> Config:
    """
    Builds a config object by applying overrides from EXPERIMENTS[exp_idx] to the default Config."""
    cfg = Config()
    overrides = EXPERIMENTS[exp_idx]
    for key, val in overrides.items():
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown config key '{key}' in EXPERIMENTS[{exp_idx}]")
        setattr(cfg, key, val)
    return cfg


_parser = argparse.ArgumentParser()
_parser.add_argument('--exp_idx', type=int, default=0)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def experiment_dir(cfg):
    folder = f"{cfg.prefix}_{cfg.name}"
    return os.path.join(cfg.output_dir, folder)

def psnr(pred, target, max_val=None):
    """PSNR between pred and target. max_val defaults to target.max()."""
    mse = torch.mean(torch.abs(pred - target) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))
    mv = target.max() if max_val is None else torch.tensor(max_val, device=pred.device)
    return 20.0 * torch.log10(mv.to(pred.device) / torch.sqrt(mse))


def _psnr_per_sample(pred, target, max_val=None):
    dims = tuple(range(1, pred.ndim))
    mse = torch.mean(torch.abs(pred - target) ** 2, dim=dims)
    if max_val is None:
        peak = target.amax(dim=dims)
    else:
        peak = torch.as_tensor(max_val, device=pred.device, dtype=pred.real.dtype)
    values = 20.0 * torch.log10(peak.to(pred.device) / torch.sqrt(mse))
    return torch.where(mse == 0, torch.full_like(values, float("inf")), values)


def config_to_dict(cfg):
    """
    Convert a config object to a dictionary.
    This is used for saving the config as JSON and logging to wandb."""

    if dataclasses.is_dataclass(cfg):
        return {k: config_to_dict(v) for k, v in dataclasses.asdict(cfg).items()}
    return cfg

def append_metrics(path, record):
    history = []
    if os.path.exists(path):
        with open(path) as f:
            history = json.load(f)
    history.append(record)
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)


# ---------------------------------------------------------------------------
# Epoch helpers
# ---------------------------------------------------------------------------

def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def _to_image_tensor(x, stats=None):
    return reconstruction_to_image_magnitude(x, stats)


def _unpack_kspace_batch(batch):
    if isinstance(batch, dict):
        return batch["kspace"], batch.get("fname")
    return batch, None


def _update_volume_errors(store, fnames, prediction, target):
    for index, fname in enumerate(fnames):
        difference = prediction[index].double() - target[index].double()
        values = store.setdefault(fname, {"sse": 0.0, "count": 0, "peak": 0.0})
        values["sse"] += difference.square().sum().item()
        values["count"] += difference.numel()
        values["peak"] = max(values["peak"], target[index].max().item())


def _mean_volume_psnr(store):
    volume_psnr = []
    for values in store.values():
        mse = values["sse"] / values["count"]
        if mse == 0:
            volume_psnr.append(float("inf"))
        elif values["peak"] > 0:
            volume_psnr.append(20.0 * math.log10(values["peak"]) - 10.0 * math.log10(mse))
    return float(np.mean(volume_psnr)) if volume_psnr else None


def _compute_losses(recon, intermediates, target, final_criterion, intermediate_criterion, loss_mode, stats=None, zf_recon=None):
    final_loss = final_criterion(recon, target, stats=stats)
    stage_losses = [intermediate_criterion(stage_out, target, stats=stats) for stage_out in intermediates]
    if stage_losses:
        intermediate_loss_sum = torch.stack(stage_losses).sum()
    else:
        intermediate_loss_sum = torch.zeros((), device=final_loss.device, dtype=final_loss.dtype)

    if loss_mode == "final_only":
        total_loss = final_loss
    elif loss_mode == "intermediate_unweighted":
        total_loss = final_loss + intermediate_loss_sum
    else:
        raise ValueError(
            f"Unknown loss_mode '{loss_mode}'. Choose from: ['final_only', 'intermediate_unweighted']"
        )

    stage_psnr_gains = []
    if zf_recon is not None and stage_losses:
        with torch.no_grad():
            gt_image = target["image"] if isinstance(target, dict) else target
            prev_img = _to_image_tensor(zf_recon, stats)
            prev_psnr = _psnr_per_sample(prev_img, gt_image)
            for stage_out in intermediates:
                stage_img = _to_image_tensor(stage_out, stats)
                curr_psnr = _psnr_per_sample(stage_img, gt_image)
                stage_psnr_gains.append(curr_psnr - prev_psnr)
                prev_psnr = curr_psnr

    return total_loss, final_loss, intermediate_loss_sum, stage_losses, stage_psnr_gains


def _set_perpendicular_weight_m(criteria, value):
    for criterion in criteria:
        if isinstance(criterion, PerpendicularLoss):
            criterion.set_current_m(value)


def _build_lambda_scheduler(cfg):
    if cfg.lambda_schedule in {"none", "hard"}:
        return None
    return LambdaScheduler(cfg.lambda_schedule, cfg.lambda_start, cfg.lambda_end, cfg.epochs)


def _init_stage_totals(num_stages):
    return [0.0 for _ in range(num_stages)]


def _num_intermediate_stages(model):
    if hasattr(model, "num_intermediate_stages"):
        return int(model.num_intermediate_stages)
    return len(model.transformers)


def _build_epoch_metrics(total_loss, final_loss, intermediate_loss_sum, total_psnr, stage_totals, sample_count, psnr_gain_totals=None):
    return {
        "total_loss": total_loss / sample_count,
        "final_loss": final_loss / sample_count,
        "intermediate_loss_sum": intermediate_loss_sum / sample_count,
        "psnr": total_psnr / sample_count,
        "stage_losses": [loss / sample_count for loss in stage_totals],
        "stage_psnr_gains": [g / sample_count for g in psnr_gain_totals] if psnr_gain_totals else [],
    }


def _build_criteria(cfg):
    loss_kwargs = {}
    if cfg.perpendicular_mag_weighting:
        loss_kwargs = {
            "magnitude_weighting": True,
            "magnitude_weight_m": cfg.perpendicular_mag_weight_m,
            "magnitude_weight_k": cfg.perpendicular_mag_weight_k,
            "magnitude_weight_p": cfg.perpendicular_mag_weight_p,
        }
    return (
        build_loss(cfg.final_loss_type, **loss_kwargs),
        build_loss(cfg.intermediate_loss_type, **loss_kwargs),
    )


def _build_optimizer(cfg, parameters):
    optimizer_type = getattr(cfg, "optimizer_type", "adam")
    if optimizer_type == "adam":
        return torch.optim.Adam(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer_type {optimizer_type!r}")


def _build_scheduler(cfg, optimizer):
    scheduler_type = getattr(cfg, "scheduler_type", "cosine")
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 1e-2
        )
    if scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma
        )
    raise ValueError(f"Unknown scheduler_type {scheduler_type!r}")


def _clip_gradients(model, max_norm):
    if max_norm is not None:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)


def _is_oom_error(error):
    oom_types = tuple(
        error_type
        for error_type in (
            getattr(torch, "OutOfMemoryError", None),
            getattr(torch.cuda, "OutOfMemoryError", None),
        )
        if isinstance(error_type, type)
    )
    message = str(error).lower()
    return isinstance(error, oom_types) or "out of memory" in message or "cudnn_status_alloc_failed" in message


def _clear_cuda_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _find_executable_batch_size(starting_batch_size, probe):
    if starting_batch_size < 1:
        raise ValueError("starting_batch_size must be at least 1")

    batch_size = starting_batch_size
    while True:
        phase(f"Testing batch size {batch_size}...")
        try:
            probe(batch_size)
        except RuntimeError as error:
            if not _is_oom_error(error):
                raise
            message = str(error)
            error.__traceback__ = None
            _clear_cuda_memory()
            if batch_size == 1:
                raise RuntimeError("Model cannot fit a batch size of 1") from error
            next_batch_size = max(1, batch_size // 2)
            phase(f"OOM at batch size {batch_size}: {message.splitlines()[0]}")
            phase(f"Retrying with batch size {next_batch_size}.")
            batch_size = next_batch_size
        else:
            _clear_cuda_memory()
            phase(f"Selected batch size: {batch_size}")
            return batch_size


def _probe_batch_candidate(cfg, dataset, batch_size, device, checkpoint=None):
    model = None
    optimizer = None
    try:
        _seed_everything(cfg.seed)
        phase(f"  Probe batch {batch_size}: building model...")
        t_build = time.time()
        model = build_model(cfg).to(device)
        phase(f"  Probe batch {batch_size}: model built in {time.time() - t_build:.1f}s")
        optimizer = _build_optimizer(cfg, model.parameters())
        final_criterion, intermediate_criterion = _build_criteria(cfg)
        if checkpoint is not None:
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])

        probe_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True,
        )
        probe_iterator = iter(probe_loader)
        mask_generator = FastMRIMaskGenerator(
            cfg.acceleration_factors,
            center_fractions=cfg.center_fractions,
            mask_type=cfg.mask_type,
        )
        model.train()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t_probe = time.time()
        probe_bar = progress_iter(
            range(cfg.batch_size_probe_steps),
            desc=f"  Probe batch {batch_size} steps",
            unit="step",
        )
        for step in probe_bar:
            t_step = time.time()
            try:
                kspace_full = next(probe_iterator)
            except StopIteration:
                probe_iterator = iter(probe_loader)
                kspace_full = next(probe_iterator)
            kspace_full = kspace_full.to(device)
            acceleration = cfg.acceleration_factors[step % len(cfg.acceleration_factors)]
            kspace_us, mask, _ = mask_generator.apply(
                kspace_full,
                acceleration,
                seed=(cfg.seed, step, int(acceleration)),
            )
            with torch.no_grad():
                model_input, dc_input, target, stats = simulate_undersampling(
                    kspace_full,
                    mask,
                    cfg.learning,
                    cfg.norm,
                    kspace_us=kspace_us,
                    robust_clip=cfg.robust_clip,
                    robust_shift=cfg.robust_shift,
                    companding_p=cfg.companding_p,
                    companding_a=cfg.companding_a,
                    companding_centering=cfg.companding_centering,
                )

            optimizer.zero_grad(set_to_none=True)
            recon, intermediates = model(
                model_input, dc_input, mask, return_intermediates=True, stats=stats
            )
            total_loss, _, _, _, _ = _compute_losses(
                recon,
                intermediates,
                target,
                final_criterion,
                intermediate_criterion,
                cfg.loss_mode,
                stats=stats,
                zf_recon=model_input,
            )
            total_loss.backward()
            _clip_gradients(model, cfg.grad_clip)
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            probe_bar.set_postfix(step_time=f"{time.time() - t_step:.1f}s")

        peak_gb = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None
        summary = (
            f"  Probe batch {batch_size}: {cfg.batch_size_probe_steps} steps OK "
            f"in {time.time() - t_probe:.1f}s"
        )
        if peak_gb is not None:
            summary += f" (peak GPU memory {peak_gb:.2f} GB)"
        phase(summary)
    finally:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        del optimizer, model


def _resolve_batch_size(cfg, dataset, device, checkpoint=None):
    if not cfg.auto_batch_size:
        return cfg.batch_size
    if device.type != "cuda":
        print(f"Automatic batch-size search skipped on {device.type}; using {cfg.batch_size}.")
        return cfg.batch_size
    if len(dataset) < 1:
        raise ValueError("Cannot search for a batch size with an empty training dataset")
    if cfg.batch_size_probe_steps < 1:
        raise ValueError("batch_size_probe_steps must be at least 1")

    starting_batch_size = min(cfg.batch_size_search_start, len(dataset))
    return _find_executable_batch_size(
        starting_batch_size,
        lambda batch_size: _probe_batch_candidate(
            cfg, dataset, batch_size, device, checkpoint=checkpoint
        ),
    )


def train_one_epoch(cfg, model, loader, accel_factors, mask_generator, optimizer,
                    final_criterion, intermediate_criterion, loss_mode, device, epoch):
    model.train()
    total_loss = 0.0
    total_final_loss = 0.0
    total_intermediate_loss_sum = 0.0
    total_psnr = 0.0
    total_samples = 0
    stage_totals = _init_stage_totals(_num_intermediate_stages(model))
    psnr_gain_totals = _init_stage_totals(_num_intermediate_stages(model))
    accel_rng = np.random.default_rng(cfg.seed + epoch)

    t_epoch = time.time()
    data_accum = 0.0
    compute_accum = 0.0
    t_wait = time.perf_counter()
    num_batches = len(loader)
    next_report = 0.1
    epoch_bar = progress_iter(loader, desc=f"Epoch {epoch + 1} train", unit="batch")
    for batch_idx, kspace_full in enumerate(epoch_bar):
        data_time = time.perf_counter() - t_wait
        t_compute = time.perf_counter()
        kspace_full = kspace_full.to(device)
        R    = accel_factors[int(accel_rng.integers(len(accel_factors)))]
        kspace_us, mask, _ = mask_generator.apply(
            kspace_full,
            R,
            seed=(cfg.seed, epoch, batch_idx, int(R)),
        )

        with torch.no_grad():
            model_input, DC_input, target, stats = simulate_undersampling(
                kspace_full,
                mask,
                cfg.learning,
                cfg.norm,
                kspace_us=kspace_us,
                robust_clip=cfg.robust_clip,
                robust_shift=cfg.robust_shift,
                companding_p=cfg.companding_p,
                companding_a=cfg.companding_a,
                companding_centering=cfg.companding_centering,
            )

        optimizer.zero_grad(set_to_none=True)
        recon, intermediates = model(
            model_input, DC_input, mask, return_intermediates=True, stats=stats
        )
        total_batch_loss, final_loss, intermediate_loss_sum, stage_losses, stage_psnr_gains = _compute_losses(
            recon, intermediates, target, final_criterion, intermediate_criterion, loss_mode, stats=stats, zf_recon=model_input
        )
        total_batch_loss.backward()
        _clip_gradients(model, cfg.grad_clip)
        optimizer.step()

        with torch.no_grad():
            gt_image = target["image"] if isinstance(target, dict) else target
            recon_mag = _to_image_tensor(recon, stats)
            batch_size = gt_image.shape[0]
            total_loss += total_batch_loss.item() * batch_size
            total_final_loss += final_loss.item() * batch_size
            total_intermediate_loss_sum += intermediate_loss_sum.item() * batch_size
            total_psnr += _psnr_per_sample(recon_mag, gt_image).sum().item()
            total_samples += batch_size
            for i, stage_loss in enumerate(stage_losses):
                stage_totals[i] += stage_loss.item() * batch_size
            for i, gain in enumerate(stage_psnr_gains):
                psnr_gain_totals[i] += gain.sum().item()

        compute_time = time.perf_counter() - t_compute
        data_accum += data_time
        compute_accum += compute_time
        t_wait = time.perf_counter()
        epoch_bar.set_postfix(
            loss=f"{total_batch_loss.item():.4f}",
            psnr=f"{total_psnr / max(total_samples, 1):.2f}",
            data=f"{data_time:.2f}s",
            comp=f"{compute_time:.2f}s",
        )

        done = batch_idx + 1
        if num_batches and done / num_batches >= next_report and next_report < 1.0:
            next_report += 0.1
            elapsed = time.time() - t_epoch
            eta = elapsed / done * (num_batches - done)
            phase(
                f"Epoch {epoch + 1} train: {100.0 * done / num_batches:.0f}% "
                f"({done}/{num_batches} batches), {elapsed / done:.2f}s/batch, "
                f"ETA {eta:.0f}s"
            )

    epoch_bar.close()
    epoch_time = time.time() - t_epoch
    phase(
        f"Epoch {epoch + 1} train done: {num_batches} batches in {epoch_time:.1f}s "
        f"({epoch_time / num_batches:.2f}s/batch; data avg {data_accum / num_batches:.2f}s, "
        f"compute avg {compute_accum / num_batches:.2f}s)"
    )

    return _build_epoch_metrics(
        total_loss, total_final_loss, total_intermediate_loss_sum, total_psnr,
        stage_totals, total_samples, psnr_gain_totals
    )


@torch.no_grad()
def validate(cfg, model, loader, accel_factors, image_size, final_criterion,
             intermediate_criterion, loss_mode, device):
    model.eval()
    total_loss = 0.0
    total_final_loss = 0.0
    total_intermediate_loss_sum = 0.0
    total_psnr = 0.0
    total_zf_psnr = 0.0
    total_samples = 0
    volume_errors = {}
    zf_volume_errors = {}
    stage_totals = _init_stage_totals(_num_intermediate_stages(model))
    psnr_gain_totals = _init_stage_totals(_num_intermediate_stages(model))

    mask_generator = FastMRIMaskGenerator(
        accel_factors,
        center_fractions=cfg.center_fractions,
        mask_type=cfg.mask_type,
    )

    t_val = time.time()
    val_bar = progress_iter(loader, desc="Validation", unit="batch")
    for batch_idx, batch in enumerate(val_bar):
        kspace_full, fnames = _unpack_kspace_batch(batch)
        kspace_full = kspace_full.to(device)
        R    = accel_factors[batch_idx % len(accel_factors)]
        kspace_us, mask, _ = mask_generator.apply(kspace_full, R, seed=(cfg.seed, batch_idx, int(R)))

        model_input, DC_input, target, stats = simulate_undersampling(
            kspace_full,
            mask,
            cfg.learning,
            cfg.norm,
            kspace_us=kspace_us,
            companding_p=cfg.companding_p,
            companding_a=cfg.companding_a,
            companding_centering=cfg.companding_centering,
        )
        recon, intermediates = model(
            model_input, DC_input, mask, return_intermediates=True, stats=stats
        )
        total_batch_loss, final_loss, intermediate_loss_sum, stage_losses, stage_psnr_gains = _compute_losses(
            recon, intermediates, target, final_criterion, intermediate_criterion, loss_mode, stats=stats, zf_recon=model_input
        )

        gt_image = target["image"] if isinstance(target, dict) else target
        recon_mag = _to_image_tensor(recon, stats)
        zf_source = model_input
        zf_mag    = _to_image_tensor(zf_source, stats)
        batch_size = gt_image.shape[0]
        total_loss += total_batch_loss.item() * batch_size
        total_final_loss += final_loss.item() * batch_size
        total_intermediate_loss_sum += intermediate_loss_sum.item() * batch_size
        total_psnr += _psnr_per_sample(recon_mag, gt_image).sum().item()
        total_zf_psnr += _psnr_per_sample(zf_mag, gt_image).sum().item()
        if fnames is not None:
            _update_volume_errors(volume_errors, fnames, recon_mag, gt_image)
            _update_volume_errors(zf_volume_errors, fnames, zf_mag, gt_image)
        total_samples += batch_size
        for i, stage_loss in enumerate(stage_losses):
            stage_totals[i] += stage_loss.item() * batch_size
        for i, gain in enumerate(stage_psnr_gains):
            psnr_gain_totals[i] += gain.sum().item()
        val_bar.set_postfix(psnr=f"{total_psnr / max(total_samples, 1):.2f}")

    val_bar.close()
    phase(f"Validation done: {len(loader)} batches in {time.time() - t_val:.1f}s")

    metrics = _build_epoch_metrics(
        total_loss, total_final_loss, total_intermediate_loss_sum, total_psnr,
        stage_totals, total_samples, psnr_gain_totals
    )
    metrics["zf_psnr"] = total_zf_psnr / total_samples
    volume_psnr = _mean_volume_psnr(volume_errors)
    zf_volume_psnr = _mean_volume_psnr(zf_volume_errors)
    metrics["volume_psnr"] = metrics["psnr"] if volume_psnr is None else volume_psnr
    metrics["zf_volume_psnr"] = metrics["zf_psnr"] if zf_volume_psnr is None else zf_volume_psnr
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parser.parse_args()
    cfg = build_cfg(args.exp_idx)
    print(f"Experiment {args.exp_idx}: {cfg.prefix}_{cfg.name}", flush=True)
    phase("Pipeline: dataset scan -> batch-size search -> wandb init -> model build -> epochs")
    _seed_everything(cfg.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- Output directory ----
    out_dir = experiment_dir(cfg)
    os.makedirs(out_dir, exist_ok=True)

    metrics_path = os.path.join(out_dir, 'metrics.json')
    config_path  = os.path.join(out_dir, 'config.json')
    best_path    = os.path.join(out_dir, 'best_model.pth')
    latest_path  = os.path.join(out_dir, 'latest.pth')

    # ---- Datasets ----
    train_data_dir, val_data_dir = resolve_data_dirs(cfg)

    def _make_dataset(data_dir, max_files=None, return_metadata=False):
        if cfg.dataset == "oasis":
            return OASISDataset(data_dir, image_size=cfg.image_size, max_files=max_files)
        return H5MRIDataset(
            data_dir,
            image_size=cfg.image_size,
            kspace_key=cfg.kspace_key,
            max_files=max_files,
            return_metadata=return_metadata,
        )

    phase(f"Loading train dataset from {train_data_dir}")
    train_ds = _make_dataset(train_data_dir, max_files=cfg.max_train_files)
    if cfg.dataset == "fastmri":
        phase("Probing one sample to determine image size...")
        sample_shape = tuple(int(x) for x in train_ds[0].shape[-2:])
        cfg.image_size = sample_shape
        phase(f"Image size: {cfg.image_size}")
    phase(f"Loading val dataset from {val_data_dir}")
    val_ds = _make_dataset(
        val_data_dir,
        max_files=cfg.max_val_files,
        return_metadata=cfg.dataset == "fastmri",
    )

    checkpoint = None
    if cfg.resume and os.path.exists(cfg.resume):
        phase(f"Loading checkpoint {cfg.resume}")
        checkpoint = torch.load(cfg.resume, map_location="cpu")

    phase("Resolving batch size...")
    if cfg.auto_batch_size and device.type == "cuda":
        phase(
            f"auto_batch_size=True: searching from {cfg.batch_size_search_start}, "
            f"{cfg.batch_size_probe_steps} full train step(s) per candidate; "
            f"each candidate also rebuilds the model, so this can take a while"
        )
    cfg.batch_size = _resolve_batch_size(cfg, train_ds, device, checkpoint=checkpoint)
    _seed_everything(cfg.seed)

    with open(config_path, 'w') as f:
        json.dump(config_to_dict(cfg), f, indent=2)

    _WANDB_PROJECT = {"fastmri": "fastMRI", "oasis": "OASIS"}
    phase("Initializing Weights & Biases...")
    t_wandb = time.time()
    wandb.init(
        project=_WANDB_PROJECT.get(cfg.dataset, "MambaCS"),
        name=f"{cfg.prefix}_{cfg.name}",
        config=config_to_dict(cfg),
    )
    phase(f"W&B ready in {time.time() - t_wandb:.1f}s")

    print(f"Experiment : {cfg.prefix}_{cfg.name}")
    print(f"Encoders   : {cfg.encoders}")
    print(f"Output dir : {out_dir}")
    print(f"Device     : {device}")
    print(f"Image size : {cfg.image_size}")
    print(f"Batch size : {cfg.batch_size}")
    print(f"Accel      : R = {cfg.acceleration_factors}  |  mask={cfg.mask_type}  |  center_fractions={cfg.center_fractions}")

    train_generator = torch.Generator().manual_seed(cfg.seed)
    val_generator = torch.Generator().manual_seed(cfg.seed)

    phase(f"Creating DataLoaders (num_workers={cfg.num_workers}, pin_memory=True)...")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True,  num_workers=cfg.num_workers,
                              pin_memory=True,
                              persistent_workers=cfg.num_workers > 0,
                              worker_init_fn=_seed_worker,
                              generator=train_generator)

    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size,
                            shuffle=False, num_workers=cfg.num_workers,
                            pin_memory=True,
                            persistent_workers=cfg.num_workers > 0,
                            worker_init_fn=_seed_worker,
                            generator=val_generator)

    file_list = getattr(val_ds, 'h5_files', None) or getattr(val_ds, 'image_files', None)
    print(f"Train dir   : {train_data_dir}")
    print(f"Val dir     : {val_data_dir}")
    if cfg.max_val_files is not None and file_list is not None:
        print(f"Val files   : {len(file_list)} capped")
    print(f"Train / Val : {len(train_ds)} / {len(val_ds)} samples")

    # ---- Model ----
    phase("Building model...")
    t_model = time.time()
    model    = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    phase(f"Model ready in {time.time() - t_model:.1f}s")
    print(f"Parameters : {n_params:,}")
    mask_generator = FastMRIMaskGenerator(
        cfg.acceleration_factors,
        center_fractions=cfg.center_fractions,
        mask_type=cfg.mask_type,
    )

    # ---- Optimiser / scheduler / loss ----
    optimizer = _build_optimizer(cfg, model.parameters())
    scheduler = _build_scheduler(cfg, optimizer)
    final_criterion, intermediate_criterion = _build_criteria(cfg)

    # ---- Resume ----
    if cfg.checkpoint_metric not in {"psnr", "volume_psnr"}:
        raise ValueError("checkpoint_metric must be 'psnr' or 'volume_psnr'")
    start_epoch = 0
    best_val_metric = float('-inf')

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        fallback_metric = (
            checkpoint.get('val_volume_psnr', float('-inf'))
            if cfg.checkpoint_metric == 'volume_psnr'
            else checkpoint.get('best_val_psnr', checkpoint.get('val_psnr', float('-inf')))
        )
        best_val_metric = checkpoint.get('best_val_metric', fallback_metric)
        print(f"Resumed from epoch {start_epoch}  ({cfg.resume})")
        checkpoint = None

    # ---- Lambda scheduler (if active) ----
    lamb_sched = _build_lambda_scheduler(cfg)
    perp_m_sched = None
    if cfg.perpendicular_mag_weighting and cfg.perpendicular_mag_weight_m_schedule != "none":
        perp_m_sched = LambdaScheduler(
            cfg.perpendicular_mag_weight_m_schedule,
            cfg.perpendicular_mag_weight_m_start,
            cfg.perpendicular_mag_weight_m_end,
            cfg.epochs,
        )

    # ---- Training loop ----
    phase(f"Starting training: epochs {start_epoch + 1} to {cfg.epochs}")
    print()
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()

        if lamb_sched is not None:
            if not hasattr(model, "set_scheduled_lamb"):
                raise ValueError(f"{cfg.model_type} does not support lambda scheduling")
            model.set_scheduled_lamb(lamb_sched.get_lambda(epoch))
        if cfg.perpendicular_mag_weighting:
            current_perp_m = cfg.perpendicular_mag_weight_m
            if perp_m_sched is not None:
                current_perp_m = perp_m_sched.get_lambda(epoch)
            _set_perpendicular_weight_m((final_criterion, intermediate_criterion), current_perp_m)

        train_metrics = train_one_epoch(
            cfg, model, train_loader, cfg.acceleration_factors, mask_generator, optimizer,
            final_criterion, intermediate_criterion, cfg.loss_mode, device, epoch
        )
        val_metrics = validate(
            cfg, model, val_loader, cfg.acceleration_factors, cfg.image_size,
            final_criterion, intermediate_criterion, cfg.loss_mode, device
        )
        scheduler.step()

        lr      = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0
        train_stage_str = " ".join(
            f"E{i+1}:{loss:.6f}" for i, loss in enumerate(train_metrics["stage_losses"])
        )
        val_stage_str = " ".join(
            f"E{i+1}:{loss:.6f}" for i, loss in enumerate(val_metrics["stage_losses"])
        )

        print(f"Epoch {epoch+1:03d}/{cfg.epochs}  |  "
              f"Train final: {train_metrics['final_loss']:.6f}  Train total: {train_metrics['total_loss']:.6f}  "
              f"Train PSNR: {train_metrics['psnr']:.2f} dB  |  "
              f"Val final: {val_metrics['final_loss']:.6f}  Val total: {val_metrics['total_loss']:.6f}  "
              f"Val PSNR: {val_metrics['psnr']:.2f} dB  (ZF: {val_metrics['zf_psnr']:.2f} dB)  |  "
              f"Val volume PSNR: {val_metrics['volume_psnr']:.2f} dB  (ZF: {val_metrics['zf_volume_psnr']:.2f} dB)  |  "
              f"Train stages [{train_stage_str}]  |  Val stages [{val_stage_str}]  |  "
              f"LR: {lr:.2e}  |  {elapsed:.1f}s")

        metrics = {
            'epoch':        epoch + 1,
            'train_final_loss': round(train_metrics['final_loss'], 6),
            'train_total_loss': round(train_metrics['total_loss'], 6),
            'train_intermediate_loss_sum': round(train_metrics['intermediate_loss_sum'], 6),
            'train_psnr':   round(train_metrics['psnr'], 4),
            'val_final_loss': round(val_metrics['final_loss'], 6),
            'val_total_loss': round(val_metrics['total_loss'], 6),
            'val_intermediate_loss_sum': round(val_metrics['intermediate_loss_sum'], 6),
            'val_psnr':     round(val_metrics['psnr'], 4),
            'val_zf_psnr':  round(val_metrics['zf_psnr'], 4),
            'val_volume_psnr': round(val_metrics['volume_psnr'], 4),
            'val_zf_volume_psnr': round(val_metrics['zf_volume_psnr'], 4),
            'lr':           lr,
            'seed':         cfg.seed,
            'time_s':       round(elapsed, 1),
        }
        for i, stage_loss in enumerate(train_metrics["stage_losses"]):
            metrics[f'train_encoder_{i+1}_loss'] = round(stage_loss, 6)
        for i, stage_loss in enumerate(val_metrics["stage_losses"]):
            metrics[f'val_encoder_{i+1}_loss'] = round(stage_loss, 6)
        for i, gain in enumerate(train_metrics["stage_psnr_gains"]):
            metrics[f'train_encoder_{i+1}_psnr_gain'] = round(gain, 6)
        for i, gain in enumerate(val_metrics["stage_psnr_gains"]):
            metrics[f'val_encoder_{i+1}_psnr_gain'] = round(gain, 6)
        if model.lamb is not False:
            for i, lv in enumerate(model.lamb):
                metrics[f'lambda_{i}'] = round(lv.item(), 6)
        elif lamb_sched is not None:
            metrics['lambda_scheduled'] = round(model.scheduled_lamb, 6)
        if cfg.perpendicular_mag_weighting:
            metrics['perpendicular_mag_weight_m'] = round(current_perp_m, 6)
        append_metrics(metrics_path, metrics)
        wandb.log(metrics, step=epoch + 1)

        selection_value = val_metrics[cfg.checkpoint_metric]
        if selection_value > best_val_metric:
            best_val_metric = selection_value
            torch.save({
                'epoch':         epoch,
                'model':         model.state_dict(),
                'optimizer':     optimizer.state_dict(),
                'scheduler':     scheduler.state_dict(),
                'checkpoint_metric': cfg.checkpoint_metric,
                'best_val_metric': best_val_metric,
                'best_val_psnr':  val_metrics['psnr'],
                'val_psnr':       val_metrics['psnr'],
                'val_volume_psnr': val_metrics['volume_psnr'],
            }, best_path)
            print(f"  -> Best model saved  ({cfg.checkpoint_metric}={best_val_metric:.4f})")

        torch.save({
            'epoch':         epoch,
            'model':         model.state_dict(),
            'optimizer':     optimizer.state_dict(),
            'scheduler':     scheduler.state_dict(),
            'checkpoint_metric': cfg.checkpoint_metric,
            'best_val_metric': best_val_metric,
            'best_val_psnr': val_metrics['psnr'],
            'val_volume_psnr': val_metrics['volume_psnr'],
        }, latest_path)

    wandb.finish()
    print(f"\nTraining complete.  Outputs saved to: {out_dir}")

if __name__ == '__main__':
    main()
