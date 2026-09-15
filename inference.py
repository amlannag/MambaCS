"""
Inference utilities for MambaCS experiments.
Provides load_experiment_model (used by notebooks) and denormalization helpers.
"""

from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid
import warnings
from zipfile import BadZipFile
import zlib

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
    "layer_no", "num_encoder_layers", "layer_norm_eps", "learned_lambda", "learning",
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
    migrated = OrderedDict()
    if hasattr(state_dict, '_metadata'):
        migrated._metadata = state_dict._metadata
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
    _VERSION = 2
    _METRIC_VERSION = 1
    _ARRAY_NAMES = tuple(f'{name}_{domain}' for name in ('ground_truth', 'zero_fill', 'reconstruction')
                         for domain in ('image', 'kspace'))
    _PNG_NAMES = tuple(f'{name}_{view}.png' for name in ('ground_truth', 'zero_fill', 'reconstruction')
                       for view in ('magnitude', 'phase', 'kspace_log_magnitude', 'kspace_phase')) + (
                           'magnitude_error.png', 'mask.png')

    @classmethod
    def for_experiment(cls, directory, config, refresh=False):
        directory = Path(directory).expanduser().resolve()
        if not any(parent.name == 'Benchmarks' and parent.parent.name == 'FASTMRI'
                   and parent.parent.parent.name == 'Experiments' for parent in directory.parents):
            return None
        return cls(directory, config, refresh=refresh)

    @staticmethod
    def _is_digest(value):
        return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)

    @staticmethod
    def _stat(path):
        stat = Path(path).stat()
        return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)

    @staticmethod
    def _jsonable(value):
        def convert(item):
            if isinstance(item, np.generic):
                return item.item()
            if isinstance(item, Path):
                return str(item)
            raise TypeError(f'Not JSON serializable: {type(item).__name__}')
        return json.loads(json.dumps(value, sort_keys=True, default=convert, allow_nan=False))

    @classmethod
    def _inference_config(cls, config):
        flat = _config_to_flat(config)
        flat.setdefault('flattening_order', 'row_major')
        if flat.get('norm') == 'kspace_companding':
            flat.setdefault('companding_centering', 'legacy')
        keys = _MODEL_KEYS | {'image_size', 'num_channels', 'kspace_key', 'norm', 'robust_clip',
                              'robust_shift', 'companding_p', 'companding_a', 'companding_centering',
                              'a', 'centering'}
        if flat.get('lambda_schedule') not in (None, False, '', 'none', 'hard', 'constant'):
            keys = keys | {'epochs'}
        return cls._jsonable({key: value for key, value in flat.items() if key in keys})

    @staticmethod
    def _read_metadata(path):
        try:
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError('expected a JSON object')
            return value
        except (OSError, ValueError, UnicodeError) as error:
            raise ValueError(f'Corrupt or missing cache metadata {path}: {error}') from error

    def _write_json(self, path, value):
        self._atomic_write(path, lambda temporary: temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False)))

    @contextmanager
    def _locked(self):
        import fcntl

        with (self.images / '.cache.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    @staticmethod
    def _mismatch(description):
        raise ValueError(f'Inference cache {description}; set REFRESH_CACHE=True in the notebook '
                         '(or pass refresh_cache=True) to explicitly replace the existing cache.')

    def __init__(self, directory, config, refresh=False):
        directory = Path(directory).expanduser().resolve()
        checkpoint = directory / 'best_model.pth'
        checkpoint_stat = self._stat(checkpoint)
        relevant_config = self._inference_config(config)
        self.refresh = bool(refresh)
        self._expected_digests = {}
        self.images = directory / 'images'
        self.images.mkdir(parents=True, exist_ok=True)
        with self._locked():
            checkpoint_stat = self._stat(checkpoint)
            index_path = self.images / 'cache_index.json'
            if index_path.exists():
                index = self._read_metadata(index_path)
                root_name = index.get('root')
                if index.get('version') != 1 or not (root_name == 'cache' or self._is_digest(root_name)):
                    raise ValueError(f'Unsupported or corrupt cache index {index_path}')
                self.root = self.images / root_name
                if not (self.root / 'manifest.json').is_file():
                    raise ValueError(f'Canonical cache manifest is missing: {self.root / "manifest.json"}; '
                                     'restore it rather than selecting a different cache.')
            else:
                candidates = [path for path in self.images.iterdir()
                              if path.is_dir() and self._is_digest(path.name) and (path / 'manifest.json').is_file()]
                first = min(candidates, key=lambda path: ((path / 'manifest.json').stat().st_mtime_ns, path.name)) if candidates else None
                self.root = first if first is not None else self.images / 'cache'
            manifest_path = self.root / 'manifest.json'
            if not index_path.exists() and manifest_path.exists():
                self._write_json(index_path, dict(version=1, root=self.root.name))
            old = self._read_metadata(manifest_path) if manifest_path.exists() else None
            if old is not None:
                if (old.get('version') not in (1, self._VERSION)
                        or not self._is_digest(old.get('checkpoint_sha256')) or not isinstance(old.get('config'), dict)):
                    raise ValueError(f'Unsupported or corrupt cache manifest {manifest_path}')
                if old['version'] == self._VERSION:
                    old_identity = dict(checkpoint_sha256=old['checkpoint_sha256'],
                                        config=self._inference_config(old['config']))
                    expected_identity = hashlib.sha256(json.dumps(old_identity, sort_keys=True).encode()).hexdigest()
                    if (not isinstance(old.get('generation'), str) or not old['generation']
                            or old.get('model_identity') != expected_identity):
                        raise ValueError(f'Unsupported or corrupt cache manifest {manifest_path}')
            if old and old.get('checkpoint_stat') == checkpoint_stat:
                checkpoint_sha256 = old['checkpoint_sha256']
            else:
                digest = hashlib.sha256()
                with checkpoint.open('rb') as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                        digest.update(chunk)
                if self._stat(checkpoint) != checkpoint_stat:
                    raise ValueError('Checkpoint changed while hashing; retry after the checkpoint writer finishes.')
                checkpoint_sha256 = digest.hexdigest()
            identity = dict(checkpoint_sha256=checkpoint_sha256, config=relevant_config)
            model_identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            changed = old is not None and (old['checkpoint_sha256'] != checkpoint_sha256
                                           or self._inference_config(old['config']) != relevant_config)
            if changed and not self.refresh:
                self._mismatch(f'checkpoint/config does not match {manifest_path}')
            generation = old['generation'] if old and old['version'] == self._VERSION and not self.refresh else uuid.uuid4().hex
            legacy_generation = old.get('legacy_generation') if old else None
            if old and old['version'] == 1 and not self.refresh:
                legacy_generation = generation
            self.metadata = dict(version=self._VERSION, **identity, checkpoint_stat=checkpoint_stat,
                                 model_identity=model_identity, generation=generation,
                                 legacy_generation=legacy_generation,
                                 code_sha256=old.get('code_sha256') if old else None)
            self.root.mkdir(parents=True, exist_ok=True)
            if old != self.metadata:
                self._write_json(manifest_path, self.metadata)
            if not index_path.exists():
                self._write_json(index_path, dict(version=1, root=self.root.name))

    def _assert_current(self):
        manifest = self._read_metadata(self.root / 'manifest.json')
        if (manifest.get('generation') != self.metadata['generation']
                or manifest.get('model_identity') != self.metadata['model_identity']):
            raise ValueError('Cache was refreshed by another process; reopen BenchmarkImageCache before continuing.')

    def _slice_slot(self, volume, slice_index, persist=False):
        if int(slice_index) != slice_index or slice_index < 0:
            raise ValueError('slice_index must be a nonnegative integer')
        base = self.root / Path(volume).name / f'slice_{slice_index:04d}'
        slot_path = base / 'slot.json'
        if slot_path.exists():
            slot = self._read_metadata(slot_path)
            name = slot.get('directory')
            if slot.get('version') != 1 or not (name == '.' or self._is_digest(name)):
                raise ValueError(f'Unsupported or corrupt slice slot {slot_path}')
        else:
            candidates = [path for path in base.iterdir() if path.is_dir() and self._is_digest(path.name)] if base.is_dir() else []
            if self._is_digest(self.root.name) and candidates and not (base / 'arrays.npz').exists():
                def first_written(path):
                    target = path / 'arrays.npz'
                    return ((target if target.exists() else path).stat().st_mtime_ns, path.name)
                name = min(candidates, key=first_written).name
            else:
                name = '.'
            if persist:
                base.mkdir(parents=True, exist_ok=True)
                self._write_json(slot_path, dict(version=1, directory=name))
        return base if name == '.' else base / name

    @staticmethod
    def _input_digest(full, undersampled, mask):
        digest = hashlib.sha256()
        for tensor in (full, undersampled, mask):
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(str((array.shape, array.dtype.str)).encode())
            digest.update(array.tobytes())
        return digest.hexdigest()

    def slice_directory(self, volume, slice_index, full, undersampled, mask):
        digest = self._input_digest(full, undersampled, mask)
        with self._locked():
            self._assert_current()
            directory = self._slice_slot(volume, slice_index, persist=True)
            self._expected_digests[directory] = digest
            self._entry(directory, expected=digest)
        return directory

    def _entry(self, directory, expected=None):
        entry_path = directory / 'entry.json'
        if entry_path.exists():
            try:
                entry = self._read_metadata(entry_path)
            except ValueError:
                return None
            if (entry.get('version') != 1 or not self._is_digest(entry.get('input_sha256'))
                    or entry.get('generation') != self.metadata['generation']
                    or entry.get('model_identity') != self.metadata['model_identity']
                    or entry.get('checkpoint_sha256') != self.metadata['checkpoint_sha256']):
                return None
        elif (self._is_digest(directory.name) and self._is_digest(self.root.name)
              and self.metadata['legacy_generation'] == self.metadata['generation']):
            entry = self._new_entry(directory.name)
        else:
            return None
        if expected is not None and entry['input_sha256'] != expected:
            if not self.refresh:
                self._mismatch(f'input/mask digest does not match slice {directory}')
            return None
        return entry

    def _new_entry(self, digest):
        return dict(version=1, input_sha256=digest, generation=self.metadata['generation'],
                    model_identity=self.metadata['model_identity'], checkpoint_sha256=self.metadata['checkpoint_sha256'])

    @staticmethod
    def _atomic_write(path, writer):
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            writer(temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _write_images(self, directory, arrays, force=False):
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
            if force or not path.is_file():
                self._atomic_write(path, lambda temporary: plt.imsave(temporary, image, cmap=cmap, vmin=vmin, vmax=vmax))

    def load(self, directory, shape):
        directory = Path(directory)
        with self._locked():
            self._assert_current()
            return self._load_slice(directory, shape, self._expected_digests.get(directory))

    def _load_slice(self, directory, shape, expected=None):
        entry = self._entry(directory, expected=expected)
        path = directory / 'arrays.npz'
        if entry is None or not path.is_file():
            return None
        try:
            if 'arrays_stat' in entry and entry['arrays_stat'] != self._stat(path):
                warnings.warn(f'Ignoring changed or invalid inference cache {path}', stacklevel=2)
                return None
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            for name in self._ARRAY_NAMES:
                array = arrays[name]
                if array.shape != tuple(shape) or not np.iscomplexobj(array) or not np.isfinite(array).all():
                    raise ValueError('Invalid cached complex slice')
            if arrays['mask'].shape != tuple(shape) or not np.isfinite(arrays['mask']).all():
                raise ValueError('Invalid cached mask')
            int(arrays['checkpoint_epoch'].item())
        except (OSError, ValueError, TypeError, KeyError, EOFError, BadZipFile, OverflowError, zlib.error) as error:
            warnings.warn(f'Ignoring incomplete or invalid inference cache {path}: {error}', stacklevel=2)
            return None
        if any(not (directory / name).is_file() for name in self._PNG_NAMES):
            self._write_images(directory, arrays)
        entry['arrays_stat'] = self._stat(path)
        entry['shape'] = list(arrays['mask'].shape)
        if not (directory / 'entry.json').is_file() or self._read_metadata(directory / 'entry.json') != entry:
            self._write_json(directory / 'entry.json', entry)
        return arrays

    def save(self, directory, full, undersampled, mask, raw_kspace, raw_image, checkpoint_epoch):
        from DcTNN.dc import ifft_2d

        directory = Path(directory)
        digest = self._input_digest(full, undersampled, mask)
        if directory not in self._expected_digests:
            raise ValueError('Call slice_directory before saving a slice.')
        if self._expected_digests[directory] != digest:
            raise ValueError('Slice inputs changed after slice_directory; resolve the slice again before saving.')
        arrays = {}
        for name, kspace, image in (('ground_truth', full, ifft_2d(full)),
                                    ('zero_fill', undersampled, ifft_2d(undersampled)),
                                    ('reconstruction', raw_kspace, raw_image)):
            arrays[f'{name}_kspace'] = kspace[0, 0].detach().cpu().numpy()
            arrays[f'{name}_image'] = image[0, 0].detach().cpu().numpy()
        arrays['mask'] = torch.broadcast_to(mask, full.shape)[0, 0].detach().cpu().numpy()
        arrays['checkpoint_epoch'] = np.asarray(checkpoint_epoch)
        with self._locked():
            self._assert_current()
            self._entry(directory, expected=digest)
            directory.mkdir(parents=True, exist_ok=True)
            self._atomic_write(directory / 'arrays.npz', lambda temporary: np.savez_compressed(temporary, **arrays))
            self._write_images(directory, arrays, force=True)
            entry = self._new_entry(digest)
            entry['arrays_stat'] = self._stat(directory / 'arrays.npz')
            entry['shape'] = list(arrays['mask'].shape)
            self._write_json(directory / 'entry.json', entry)

    def volume_context(self, path, file_index, settings, mask_seed):
        path = Path(path).expanduser().resolve()
        return self._jsonable(dict(version=1, metric_contract_version=self._METRIC_VERSION,
                                   generation=self.metadata['generation'], model_identity=self.metadata['model_identity'],
                                   file=dict(name=path.name, path=str(path), **self._stat(path)),
                                   file_index=file_index, mask_seed=mask_seed, settings=settings))

    @staticmethod
    def _volume_result(result):
        if not isinstance(result, dict) or not {
                'slice_count', 'checkpoint_epoch', 'h5_max', 'metrics', 'zero_fill_metrics'} <= result.keys():
            raise ValueError('Volume result requires slice_count, checkpoint_epoch, h5_max, metrics and zero_fill_metrics')
        count = result['slice_count']
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count <= 0:
            raise ValueError('Volume slice_count must be a positive integer')
        epoch = result['checkpoint_epoch']
        if isinstance(epoch, bool) or not isinstance(epoch, (int, np.integer)):
            raise ValueError('Volume checkpoint_epoch must be an integer')
        value = dict(slice_count=int(count), checkpoint_epoch=int(epoch), h5_max=float(result['h5_max']))
        if not np.isfinite(value['h5_max']) or value['h5_max'] <= 0:
            raise ValueError('Volume h5_max must be positive and finite')
        for field in ('metrics', 'zero_fill_metrics'):
            metrics = result[field]
            if not isinstance(metrics, dict) or len(metrics) != 4:
                raise ValueError(f'{field} must contain the four benchmark metrics')
            value[field] = {name: float(metric) for name, metric in metrics.items()}
            if any(np.isnan(metric) for metric in value[field].values()):
                raise ValueError(f'{field} contains NaN')
        if value['metrics'].keys() != value['zero_fill_metrics'].keys():
            raise ValueError('Reconstruction and zero-fill metric names must agree')
        return value

    def _slice_record(self, volume, slice_index):
        try:
            directory = self._slice_slot(volume, slice_index)
            entry = self._entry(directory)
            if entry is None:
                return None
            arrays_stat = self._stat(directory / 'arrays.npz')
            if arrays_stat['size'] <= 0 or entry.get('arrays_stat') != arrays_stat:
                return None
            return dict(directory=str(directory.relative_to(self.root)), arrays_stat=arrays_stat,
                        entry_stat=self._stat(directory / 'entry.json'), input_sha256=entry['input_sha256'])
        except (OSError, ValueError):
            return None

    def load_volume(self, path, file_index, settings, mask_seed):
        path = Path(path)
        with self._locked():
            self._assert_current()
            summary_path = self.root / path.name / 'volume_metrics.json'
            if not summary_path.is_file():
                return None
            try:
                summary = self._read_metadata(summary_path)
                context = summary['context']
                if summary.get('version') != 1 or not isinstance(context, dict):
                    return None
                current = self.volume_context(path, file_index, settings, mask_seed)
                if (context.get('generation') != current['generation']
                        or context.get('model_identity') != current['model_identity']
                        or context.get('metric_contract_version') != self._METRIC_VERSION):
                    return None
                if (not {'version', 'file', 'file_index', 'mask_seed', 'settings'} <= context.keys()
                        or context['version'] != 1 or not isinstance(context['file'], dict)
                        or not {'name', 'path', 'size', 'mtime_ns', 'ctime_ns'} <= context['file'].keys()):
                    return None
            except (ValueError, TypeError, KeyError, OSError, OverflowError):
                return None
            if context != current:
                if not self.refresh:
                    self._mismatch(f'data/mask/settings do not match volume {path.name}')
                return None
            try:
                result = self._volume_result(summary['result'])
                records = summary['slices']
                if not isinstance(records, list) or len(records) != result['slice_count']:
                    return None
            except (ValueError, TypeError, KeyError, OverflowError):
                return None
            for slice_index, record in enumerate(records):
                if not isinstance(record, dict) or self._slice_record(path.name, slice_index) != record:
                    return None
            for record in records:
                directory = self.root / record['directory']
                if any(not (directory / name).is_file() for name in self._PNG_NAMES):
                    entry = self._entry(directory)
                    shape = entry.get('shape') if entry else None
                    if not isinstance(shape, list) or len(shape) != 2:
                        return None
                    if self._load_slice(directory, shape) is None:
                        return None
            return result

    def save_volume(self, path, file_index, settings, mask_seed, result):
        path = Path(path)
        result = self._volume_result(result)
        with self._locked():
            self._assert_current()
            context = self.volume_context(path, file_index, settings, mask_seed)
            records = [self._slice_record(path.name, index) for index in range(result['slice_count'])]
            if any(record is None for record in records):
                raise ValueError(f'Cannot save incomplete volume cache for {path.name}; load or save every slice first.')
            summary_path = self.root / path.name / 'volume_metrics.json'
            if summary_path.is_file():
                try:
                    old_context = self._read_metadata(summary_path).get('context', {})
                except ValueError:
                    old_context = {}
                if (isinstance(old_context, dict) and old_context.get('generation') == context['generation']
                        and old_context.get('model_identity') == context['model_identity']
                        and old_context != context and not self.refresh):
                    self._mismatch(f'data/mask/settings do not match volume {path.name}')
            serialized = dict(result)
            for field in ('metrics', 'zero_fill_metrics'):
                serialized[field] = {name: metric if np.isfinite(metric) else str(metric)
                                     for name, metric in result[field].items()}
            self._write_json(summary_path, dict(version=1, context=context, result=serialized, slices=records))

    def read_slice(self, volume, slice_index, shape):
        with self._locked():
            self._assert_current()
            try:
                directory = self._slice_slot(volume, slice_index)
            except (OSError, ValueError):
                return None
            return self._load_slice(directory, shape)
