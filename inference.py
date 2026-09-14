"""
Inference utilities for MambaCS experiments.
Provides load_experiment_model (used by notebooks) and denormalization helpers.
"""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import warnings
from zipfile import BadZipFile

import numpy as np
import torch

from config import Config
from train_utils import build_model, validate_resume_fixed_apt_config
from normalizer import invert_normalization, reconstruction_to_image_magnitude


# Fields that belong in cfg['data'] vs cfg['model'] for notebook consumers.
_DATA_KEYS = {
    "dataset", "data_dir", "val_data_dir", "kspace_key", "image_size",
    "num_channels", "acceleration_factors", "center_fractions", "mask_type",
    "val_fraction", "seed", "max_train_files", "max_val_files", "norm", "robust_clip", "robust_shift",
    "companding_p", "companding_a", "companding_centering",
}
_MODEL_KEYS = {
    "model_type", "encoders", "patch_size", "axial_row_stride", "nhead_patch", "nhead_axial",
    "layer_no", "num_encoder_layers", "learned_lambda", "learning",
    "reconformer_num_ch", "reconformer_num_iter", "reconformer_down_scales",
    "reconformer_num_heads", "reconformer_depths", "reconformer_window_sizes",
    "reconformer_mlp_ratio", "reconformer_resi_connection", "reconformer_use_checkpoint",
    "lambda_schedule", "lambda_start", "lambda_end",
    "pos_emb_type", "attn_type", "rope_theta", "rope_mixed_rotate",
    "mask_vertical_attn", "ffn_sharing", "flattening_order",
    "apt_layout", "apt_embed_dim", "apt_rope_ref_grid", "apt_use_abs_pos_emb",
}


def _config_to_flat(config: dict) -> dict:
    if not isinstance(config, dict):
        raise ValueError("Saved configuration must be a dictionary")
    sections = ("train", "data", "model")
    flat = {key: value for key, value in config.items() if key not in sections}
    for name in sections:
        section = config.get(name)
        if section is None:
            continue
        if not isinstance(section, dict):
            raise ValueError(f"Configuration section {name!r} must be a dictionary")
        for key, value in section.items():
            if key in flat and json.dumps(flat[key], sort_keys=True) != json.dumps(value, sort_keys=True):
                raise ValueError(f"Conflicting configuration values for {key!r} in section {name!r}")
            flat[key] = value
    return flat


def _flat_to_nested(flat: dict) -> dict:
    """Split a flat config dict into {'data': {...}, 'model': {...}, 'train': {...}}."""
    flat = _config_to_flat(flat)
    flat.setdefault("flattening_order", "row_major")
    data, model, train = {}, {}, {}
    for k, v in flat.items():
        if k in _DATA_KEYS:
            data[k] = v
        elif k in _MODEL_KEYS:
            model[k] = v
        else:
            train[k] = v
    return {"data": data, "model": model, "train": train}


def _flat_to_cfg(flat: dict) -> Config:
    """Reconstruct a Config dataclass from the flat dict saved by train.py."""
    flat = _config_to_flat(flat)
    cfg = Config()
    if flat.get("norm") == "kspace_companding" and "companding_centering" not in flat:
        cfg.companding_centering = "legacy"
    for k, v in flat.items():
        if not hasattr(cfg, k):
            continue
        if k in {"image_size", "patch_size"} and type(v) is int:
            v = (v, v)
        if k in {
            "image_size", "patch_size", "reconformer_num_ch", "reconformer_down_scales",
            "reconformer_num_heads", "reconformer_depths", "reconformer_window_sizes",
            "reconformer_use_checkpoint", "apt_rope_ref_grid",
        } and isinstance(v, list):
            v = tuple(v)
        setattr(cfg, k, v)
    return cfg


def _migrate_complex_layernorm(state_dict: dict) -> dict:
    """
    Migrate checkpoints saved with the old ComplexLayerNorm (single complex gamma)
    to the new 2x2 covariance form (gamma_rr, gamma_ii, gamma_ri).

    Old form: gamma = complex tensor, applied as complex multiplication
        real_out = whitened.real * gamma.real - whitened.imag * gamma.imag
        imag_out = whitened.real * gamma.imag + whitened.imag * gamma.real
    New form: 2x2 matrix [[gamma_rr, gamma_ri], [gamma_ri, gamma_ii]]
    Mapping:  gamma_rr = gamma.real, gamma_ii = gamma.real, gamma_ri = -gamma.imag
              (approximation — whitening also changed from scalar to 2x2 covariance)
    """
    migrated = {}
    for k, v in state_dict.items():
        if k.endswith(".gamma") and torch.is_complex(v):
            base = k[: -len(".gamma")]
            migrated[base + ".gamma_rr"] = v.real.clone()
            migrated[base + ".gamma_ii"] = v.real.clone()
            migrated[base + ".gamma_ri"] = -v.imag.clone()
        else:
            migrated[k] = v
    return migrated


def load_experiment_model(exp_dir: str, device=None):
    """
    Load a trained experiment from an Experiments/<name>/ directory.

    Returns a dict with:
        'model':      the loaded nn.Module in eval mode
        'config':     nested dict {'data': {...}, 'model': {...}, 'train': {...}}
        'checkpoint': raw checkpoint metadata (epoch, val_psnr, best_val_psnr)
        'cfg':        Config dataclass (for code that prefers attribute access)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = os.path.join(exp_dir, "config.json")
    ckpt_path = os.path.join(exp_dir, "best_model.pth")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True) if os.path.exists(ckpt_path) else None
    if os.path.exists(config_path):
        with open(config_path) as f:
            flat = _config_to_flat(json.load(f))
    elif ckpt is not None and isinstance(ckpt.get("config"), dict):
        flat = _config_to_flat(ckpt["config"])
    else:
        raise FileNotFoundError(f"No config.json or checkpoint-embedded configuration found in {exp_dir}")
    if flat.get("norm") == "kspace_companding" and "companding_centering" not in flat:
        flat["companding_centering"] = "legacy"

    cfg = _flat_to_cfg(flat)
    if ckpt is not None and isinstance(ckpt.get("config"), dict):
        validate_resume_fixed_apt_config(cfg, ckpt["config"])
    model = build_model(cfg).to(device)

    ckpt_meta = {}
    if ckpt is not None:
        sd = _migrate_complex_layernorm(ckpt["model"])
        missing, unexpected = model.load_state_dict(sd, strict="fixed_apt" in cfg.encoders)
        if missing or unexpected:
            print(f"  [WARN] {os.path.basename(exp_dir)}: "
                  f"{len(missing)} missing, {len(unexpected)} unexpected keys "
                  f"(ComplexLayerNorm migration applied)")
        ckpt_meta = {k: v for k, v in ckpt.items() if k != "model"}
    else:
        print(f"  WARNING — no checkpoint found at {ckpt_path}, using random weights")

    model.eval()

    return {
        "model":      model,
        "config":     _flat_to_nested(flat),
        "checkpoint": ckpt_meta,
        "cfg":        cfg,
    }


def _denormalize_image(recon: torch.Tensor, stats: dict) -> torch.Tensor:
    """
    Restore a reconstructed tensor toward original units.
    For normalized k-space modes this returns original-scale complex k-space.
    For z-score stats it preserves the previous real/complex affine restoration.
    """
    if recon.is_complex() and stats and stats.get("normalization") in {
        "kspace_companding",
        "log_kspace",
        "fastmri_magnitude",
        "reconformer",
        "robust_shifted",
    }:
        return invert_normalization(recon, stats)
    if not stats or "mean_r" not in stats:
        return recon
    mean_r, std_r = stats["mean_r"], stats["std_r"]
    mean_i, std_i = stats["mean_i"], stats["std_i"]
    if recon.is_complex():
        return torch.complex(
            recon.real * std_r + mean_r,
            recon.imag * std_i + mean_i,
        )
    return recon * std_r + mean_r


def to_image_magnitude(recon: torch.Tensor, stats: dict | None = None) -> torch.Tensor:
    """Convert a reconstruction into image magnitude in its configured prediction domain."""
    return reconstruction_to_image_magnitude(recon, stats)


class BenchmarkImageCache:
    @classmethod
    def for_experiment(cls, directory, config):
        directory = Path(directory).expanduser().resolve()
        if not any(parent.name == 'Benchmarks' and parent.parent.name == 'FASTMRI'
                   and parent.parent.parent.name == 'Experiments' for parent in directory.parents):
            return None
        return cls(directory, config)

    def __init__(self, directory, config):
        checkpoint = directory / 'best_model.pth'
        digest = hashlib.sha256()
        with checkpoint.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        root = Path(__file__).resolve().parent
        sources = [root / name for name in ('inference.py', 'normalizer.py', 'dataset.py', 'train_utils.py', 'train.py',
                                            'notebooks/notebook_inference.py') if (root / name).is_file()]
        sources += sorted((root / 'DcTNN').glob('*.py')) + sorted((root / 'ReconFormer').glob('*.py'))
        code_digest = hashlib.sha256()
        for source in sources:
            code_digest.update(str(source.relative_to(root)).encode())
            code_digest.update(source.read_bytes())
        self.metadata = dict(version=1, checkpoint_sha256=digest.hexdigest(), config=_config_to_flat(config),
                             code_sha256=code_digest.hexdigest())
        serialized = json.dumps(self.metadata, sort_keys=True)
        self.root = directory / 'images' / hashlib.sha256(serialized.encode()).hexdigest()

    def slice_directory(self, volume, slice_index, full, undersampled, mask):
        digest = hashlib.sha256()
        for tensor in (full, undersampled, mask):
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(str((array.shape, array.dtype.str)).encode())
            digest.update(array.tobytes())
        return self.root / Path(volume).name / f'slice_{slice_index:04d}' / digest.hexdigest()

    @staticmethod
    def _atomic_write(path, writer):
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            writer(temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _write_images(self, directory, arrays):
        import matplotlib.pyplot as plt

        gt_magnitude = np.abs(arrays['ground_truth_image'])
        gt_log_kspace = np.log1p(np.abs(arrays['ground_truth_kspace']))
        magnitude_max = max(float(np.percentile(gt_magnitude, 99.5)), np.finfo(float).eps)
        kspace_max = max(float(np.percentile(gt_log_kspace, 99.5)), np.finfo(float).eps)
        images = {}
        for name in ('ground_truth', 'zero_fill', 'reconstruction'):
            image, kspace = arrays[f'{name}_image'], arrays[f'{name}_kspace']
            images[f'{name}_magnitude'] = (np.abs(image), 'gray', 0, magnitude_max)
            images[f'{name}_phase'] = (np.angle(image), 'twilight', -np.pi, np.pi)
            images[f'{name}_kspace_log_magnitude'] = (np.log1p(np.abs(kspace)), 'viridis', 0, kspace_max)
            images[f'{name}_kspace_phase'] = (np.angle(kspace), 'twilight', -np.pi, np.pi)
        error = np.abs(np.abs(arrays['reconstruction_image']) - gt_magnitude)
        images['magnitude_error'] = (error, 'magma', 0, max(float(np.percentile(error, 99.5)), np.finfo(float).eps))
        images['mask'] = (arrays['mask'], 'gray', 0, 1)
        for name, (image, cmap, vmin, vmax) in images.items():
            path = directory / f'{name}.png'
            if not path.is_file():
                self._atomic_write(path, lambda temporary: plt.imsave(temporary, image, cmap=cmap, vmin=vmin, vmax=vmax))

    def load(self, directory, shape):
        path = directory / 'arrays.npz'
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            for name in ('ground_truth', 'zero_fill', 'reconstruction'):
                for domain in ('image', 'kspace'):
                    array = arrays[f'{name}_{domain}']
                    if array.shape != shape or not np.iscomplexobj(array) or not np.isfinite(array).all():
                        raise ValueError('Invalid cached complex slice')
            if arrays['mask'].shape != shape or not np.isfinite(arrays['mask']).all():
                raise ValueError('Invalid cached mask')
            int(arrays['checkpoint_epoch'].item())
        except (OSError, ValueError, KeyError, EOFError, BadZipFile) as error:
            warnings.warn(f'Ignoring incomplete or invalid inference cache {path}: {error}', stacklevel=2)
            return None
        self._write_images(directory, arrays)
        return arrays

    def save(self, directory, full, undersampled, mask, raw_kspace, raw_image, checkpoint_epoch):
        from DcTNN.dc import ifft_2d

        directory.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for name, kspace, image in (('ground_truth', full, ifft_2d(full)),
                                    ('zero_fill', undersampled, ifft_2d(undersampled)),
                                    ('reconstruction', raw_kspace, raw_image)):
            arrays[f'{name}_kspace'] = kspace[0, 0].detach().cpu().numpy()
            arrays[f'{name}_image'] = image[0, 0].detach().cpu().numpy()
        arrays['mask'] = torch.broadcast_to(mask, full.shape)[0, 0].detach().cpu().numpy()
        arrays['checkpoint_epoch'] = np.asarray(checkpoint_epoch)
        manifest = self.root / 'manifest.json'
        if not manifest.is_file():
            self._atomic_write(manifest, lambda temporary: temporary.write_text(json.dumps(self.metadata, indent=2)))
        self._write_images(directory, arrays)
        self._atomic_write(directory / 'arrays.npz', lambda temporary: np.savez_compressed(temporary, **arrays))


def _json_metric_values(value):
    if isinstance(value, dict):
        return {key: _json_metric_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_metric_values(item) for item in value]
    return str(value) if isinstance(value, float) and not np.isfinite(value) else value


def _reconformer_metric_rows(report):
    rows = []
    for record in report['per_volume']:
        common = {'HDF5 volume': record['HDF5 volume'], 'Slices': record['Slices']}
        rows.append(dict(common, Method='ReconFormer', **{name: record[name] for name in report['metric_names']}))
        rows.append(dict(common, Method='Zero-fill', **record['zero_fill']))
    summary = [dict(Method='ReconFormer', **report['means']), dict(Method='Zero-fill', **report['zero_fill_means'])]
    return rows, summary


def _write_reconformer_metrics(directory, report):
    import csv

    def write_csv(path, fields, rows):
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    names = report['metric_names']
    rows, summary = _reconformer_metric_rows(report)
    BenchmarkImageCache._atomic_write(
        directory / 'metrics.json',
        lambda path: path.write_text(json.dumps(_json_metric_values(report), indent=2, allow_nan=False)),
    )
    BenchmarkImageCache._atomic_write(
        directory / 'per_volume.csv', lambda path: write_csv(path, ['HDF5 volume', 'Slices', 'Method', *names], rows),
    )
    BenchmarkImageCache._atomic_write(
        directory / 'summary.csv', lambda path: write_csv(path, ['Method', *names], summary),
    )


def _log_reconformer_metrics(run, report):
    metrics = {f'inference/{name}': value for name, value in report['means'].items()}
    metrics.update({f'zero_fill/{name}': value for name, value in report['zero_fill_means'].items()})
    metrics.update({
        'inference/complete': report['complete'], 'progress/volumes': report['volume_count'],
        'progress/total_volumes': report['total_volumes'], 'progress/slices': report['slice_count'],
        'progress/cached_slices': report['cached_slices'], 'progress/computed_slices': report['computed_slices'],
        'progress/elapsed_seconds': report.get('elapsed_seconds', 0.0),
    })
    if report['per_volume']:
        volume = report['per_volume'][-1]
        metrics.update({'volume/name': volume['HDF5 volume'], 'volume/slices': volume['Slices']})
        metrics.update({f'volume/reconformer/{name}': volume[name] for name in report['metric_names']})
        metrics.update({f'volume/zero_fill/{name}': value for name, value in volume['zero_fill'].items()})
    metrics = _json_metric_values(metrics)
    run.log(metrics, step=report['volume_count'])
    run.summary.update(metrics)


def _log_reconformer_tables(run, report):
    import wandb

    if not report['per_volume']:
        return
    rows, summary = _reconformer_metric_rows(report)
    tables = {}
    for name, fields, records in (
        ('per_volume', ['HDF5 volume', 'Slices', 'Method', *report['metric_names']], rows),
        ('summary', ['Method', *report['metric_names']], summary),
    ):
        tables[f'inference/{name}'] = wandb.Table(
            columns=fields, data=_json_metric_values([[record[field] for field in fields] for record in records]),
            allow_mixed_types=True,
        )
    run.log(tables)


@torch.inference_mode()
def evaluate_reconformer(experiment_dir, data_dir=None, output_dir=None, device=None, mask_seed=None,
                         wandb_project='fastMRI', wandb_entity=None, wandb_run_name=None):
    from datetime import datetime, timezone
    import time
    import h5py
    import wandb
    from DcTNN.dc import ifft_2d
    from dataset import prepare_fastmri_kspace
    from train import (_FINAL_VAL_METRICS, _new_volume_accumulator, _update_volume_accumulator,
                       _finalize_volume_metrics, _run_validation_slice)
    from train_utils import FastMRIMaskGenerator, resolve_data_dirs

    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA/ROCm GPU requested, but torch.cuda.is_available() is false')
    experiment_dir = Path(experiment_dir).expanduser().resolve()
    checkpoint_path = experiment_dir / 'best_model.pth'
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Missing trained checkpoint: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    config_path = experiment_dir / 'config.json'
    if config_path.is_file():
        with config_path.open() as handle:
            flat = _config_to_flat(json.load(handle))
    elif isinstance(checkpoint.get('config'), dict):
        flat = _config_to_flat(checkpoint['config'])
    else:
        raise FileNotFoundError(f'No config.json or checkpoint-embedded configuration in {experiment_dir}')
    cfg = _flat_to_cfg(flat)
    if cfg.model_type != 'reconformer' or cfg.dataset != 'fastmri':
        raise ValueError('This inference job requires a fastMRI ReconFormer experiment')
    if cfg.learning != 'complex_image' or cfg.norm != 'reconformer':
        raise ValueError('ReconFormer requires complex_image learning and reconformer normalization')
    if len(cfg.acceleration_factors) != 1:
        raise ValueError('Evaluation requires exactly one configured acceleration factor')
    data_dir = Path(data_dir or resolve_data_dirs(cfg)[1]).expanduser().resolve()
    files = sorted(data_dir.glob('*.h5'))
    if not files:
        raise FileNotFoundError(f'No HDF5 validation volumes found in {data_dir}')
    mask_seed = int(cfg.seed if mask_seed is None else mask_seed)
    acceleration = int(cfg.acceleration_factors[0])
    mask_generator = FastMRIMaskGenerator(cfg.acceleration_factors, cfg.center_fractions, cfg.mask_type)
    cache = BenchmarkImageCache.for_experiment(experiment_dir, flat)
    if cache is not None:
        checkpoint_digest = cache.metadata['checkpoint_sha256']
    else:
        digest = hashlib.sha256()
        with checkpoint_path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        checkpoint_digest = digest.hexdigest()
    started = datetime.now(timezone.utc)
    started_clock = time.monotonic()
    directory = Path(output_dir).expanduser().resolve() if output_dir is not None else None
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=False)
    checkpoint_epoch = checkpoint.get('epoch')
    report = dict(
        complete=False, started_at=started.isoformat(), output_dir=str(directory) if directory is not None else None,
        experiment_dir=str(experiment_dir), checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_digest,
        checkpoint_epoch=None if checkpoint_epoch is None else int(checkpoint_epoch) + 1,
        data_dir=str(data_dir), device=str(device), mask_seed=mask_seed,
        image_size=list(cfg.image_size), acceleration=acceleration, center_fractions=cfg.center_fractions,
        mask_type=cfg.mask_type, normalization=cfg.norm, training_loss=cfg.final_loss_type,
        aggregation='Equal-weight mean of per-volume metrics; all slices and all k-space locations',
        metric_names=list(_FINAL_VAL_METRICS),
        metric_definitions={
            'Image Mag PSNR': 'Volume magnitude MSE, using HDF5 max as peak; dB',
            'Image Phase Loss': 'Unweighted mean squared wrapped image phase difference; radians squared',
            'K-space L1': 'Mean absolute complex k-space difference in raw units',
            'K-space Phase Loss': 'Unweighted mean squared wrapped k-space phase difference; radians squared',
        },
        nonfinite_json_values='Nonfinite values such as perfect-reconstruction PSNR are strings (inf)',
        images_dir=str(cache.root) if cache is not None else None,
        total_volumes=len(files), volume_count=0, slice_count=0, computed_slices=0, cached_slices=0,
        means={}, zero_fill_means={}, per_volume=[],
    )
    result_keys = {'complete', 'volume_count', 'slice_count', 'computed_slices', 'cached_slices',
                   'means', 'zero_fill_means', 'per_volume'}
    with wandb.init(
        project=wandb_project, entity=wandb_entity, job_type='inference', mode='online',
        name=wandb_run_name or f"{cfg.prefix}_{cfg.name}_inference_{started.strftime('%Y%m%dT%H%M%S')}",
        group=f'{cfg.prefix}_{cfg.name}',
        config=_json_metric_values({key: value for key, value in report.items() if key not in result_keys}),
    ) as run:
        report['wandb_url'] = run.url
        _log_reconformer_metrics(run, report)
        if directory is not None:
            _write_reconformer_metrics(directory, report)
        print(f'Evaluating best checkpoint over ALL {len(files)} validation volumes on {device}', flush=True)
        print(f'Weights & Biases run: {run.url}', flush=True)
        model = None
        try:
            for file_index, path in enumerate(files):
                accumulator, zero_fill = _new_volume_accumulator(), _new_volume_accumulator()
                with h5py.File(path, 'r') as handle:
                    if 'max' not in handle.attrs:
                        raise ValueError(f"Missing HDF5 'max' attribute in {path}")
                    peak = float(handle.attrs['max'])
                    if not np.isfinite(peak) or peak <= 0:
                        raise ValueError(f'HDF5 max must be positive and finite in {path}')
                    dataset = handle[cfg.kspace_key]
                    if dataset.ndim != 3 or not np.issubdtype(dataset.dtype, np.complexfloating) or not dataset.shape[0]:
                        raise ValueError(f'Expected nonempty complex [slices, H, W] k-space in {path}')
                    slice_count = int(dataset.shape[0])
                    for slice_index in range(slice_count):
                        full = prepare_fastmri_kspace(torch.as_tensor(dataset[slice_index], dtype=torch.complex64), cfg.image_size)
                        full = full[None, None].to(device)
                        if not torch.isfinite(full).all():
                            raise ValueError(f'Nonfinite k-space in {path}, slice {slice_index}')
                        undersampled, mask, _ = mask_generator.apply(
                            full, acceleration, seed=(mask_seed, file_index, slice_index, acceleration)
                        )
                        cached = slice_directory = None
                        if cache is not None:
                            slice_directory = cache.slice_directory(path.name, slice_index, full, undersampled, mask)
                            cached = cache.load(slice_directory, tuple(full.shape[-2:]))
                        if cached is not None:
                            pred_kspace, pred_image = (torch.from_numpy(cached[f'reconstruction_{domain}']).to(device)[None, None]
                                                       for domain in ('kspace', 'image'))
                            report['cached_slices'] += 1
                        else:
                            if model is None:
                                model = build_model(cfg)
                                model.load_state_dict(checkpoint['model'], strict=True)
                                model.to(device).eval()
                            pred_kspace, pred_image = _run_validation_slice(cfg, model, full, undersampled, mask)
                            if not torch.isfinite(pred_kspace).all() or not torch.isfinite(pred_image).all():
                                raise ValueError(f'Nonfinite reconstruction in {path}, slice {slice_index}')
                            if cache is not None:
                                cache.save(slice_directory, full, undersampled, mask, pred_kspace, pred_image,
                                           int(checkpoint_epoch) if checkpoint_epoch is not None else cfg.epochs - 1)
                            report['computed_slices'] += 1
                        gt_image = ifft_2d(full)
                        _update_volume_accumulator(accumulator, gt_image, pred_image, full, pred_kspace)
                        _update_volume_accumulator(zero_fill, gt_image, ifft_2d(undersampled), full, undersampled)
                report['per_volume'].append({
                    'HDF5 volume': path.name, 'Slices': slice_count,
                    **_finalize_volume_metrics(accumulator, peak), 'zero_fill': _finalize_volume_metrics(zero_fill, peak),
                })
                report['volume_count'] += 1
                report['slice_count'] += slice_count
                report['means'] = {name: float(np.mean([row[name] for row in report['per_volume']])) for name in _FINAL_VAL_METRICS}
                report['zero_fill_means'] = {name: float(np.mean([row['zero_fill'][name] for row in report['per_volume']]))
                                            for name in _FINAL_VAL_METRICS}
                report['elapsed_seconds'] = time.monotonic() - started_clock
                report['complete'] = report['volume_count'] == len(files)
                _log_reconformer_metrics(run, report)
                if directory is not None:
                    _write_reconformer_metrics(directory, report)
                print(f"[{file_index + 1}/{len(files)}] {path.name}: {slice_count} slices; "
                      f"cached={report['cached_slices']}, computed={report['computed_slices']}", flush=True)
        finally:
            _log_reconformer_tables(run, report)
    print(f"{'Metric':<24} {'ReconFormer':>14} {'Zero-fill':>14}")
    for name in _FINAL_VAL_METRICS:
        print(f"{name:<24} {report['means'][name]:>14.6g} {report['zero_fill_means'][name]:>14.6g}")
    print(f"Metrics uploaded to Weights & Biases: {report['wandb_url']}", flush=True)
    if directory is not None:
        print(f'Local copies saved in {directory}', flush=True)
    return report


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate an HPC-trained ReconFormer best checkpoint on every fastMRI validation volume.')
    parser.add_argument('--experiment-dir', type=Path, required=True, help='Folder containing best_model.pth and its saved config.json')
    parser.add_argument('--data-dir', type=Path, help='Validation HDF5 directory; defaults to saved val_data_dir or the configured HPC path')
    parser.add_argument('--output-dir', type=Path, help='Optional new directory for local JSON/CSV copies; metrics upload to W&B by default')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto', help='CUDA also selects AMD GPUs with a ROCm PyTorch build')
    parser.add_argument('--mask-seed', type=int, help='Override the saved mask seed while retaining deterministic per-volume/per-slice masks')
    parser.add_argument('--wandb-project', default='fastMRI', help='W&B project for the inference run (default: fastMRI)')
    parser.add_argument('--wandb-entity', help='Optional W&B team or username; otherwise use the configured account')
    parser.add_argument('--wandb-run-name', help='Optional name for this inference run')
    args = parser.parse_args(argv)
    evaluate_reconformer(args.experiment_dir, args.data_dir, args.output_dir,
                         None if args.device == 'auto' else args.device, args.mask_seed,
                         wandb_project=args.wandb_project, wandb_entity=args.wandb_entity, wandb_run_name=args.wandb_run_name)


if __name__ == '__main__':
    main()
