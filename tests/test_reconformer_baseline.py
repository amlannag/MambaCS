import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault(
    "wandb",
    types.SimpleNamespace(init=lambda *a, **k: None, log=lambda *a, **k: None, finish=lambda *a, **k: None),
)

from config import Config
from dataset import H5MRIDataset, centered_fft2, centered_ifft2
from DcTNN.loss import ReconFormerMagnitudeL1Loss, build_loss
from inference import load_experiment_model
from normalizer import model_output_to_raw_kspace
from ReconFormer import ReconFormerBaseline, centered_fft2 as reconformer_fft2
from train import _build_optimizer, _mean_volume_psnr, _update_volume_errors, build_cfg, train_one_epoch
from train_utils import FastMRIMaskGenerator, build_model, simulate_undersampling


class ReconFormerBaselineTest(unittest.TestCase):
    def test_experiment_builds_released_reconformer_configuration(self):
        cfg = build_cfg(1)
        model = build_model(cfg)

        self.assertIsInstance(model, ReconFormerBaseline)
        self.assertEqual(cfg.learning, "complex_image")
        self.assertEqual(cfg.norm, "reconformer")
        self.assertEqual(cfg.final_loss_type, "reconformer_l1")
        self.assertEqual(cfg.mask_type, "random")
        self.assertEqual(cfg.checkpoint_metric, "volume_psnr")
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 1_141_299)

    def test_reconformer_normalization_matches_zero_fill_mean_magnitude(self):
        image = torch.tensor(
            [[[[1.0 + 1.0j, 2.0 - 1.0j], [0.5 + 0.25j, -1.0 + 0.5j]]]],
            dtype=torch.complex64,
        )
        kspace = centered_fft2(image)
        mask = torch.tensor([[[[1.0, 0.0]]]])
        kspace_us = kspace * mask

        model_input, dc_input, target, stats = simulate_undersampling(
            kspace,
            mask,
            learning="complex_image",
            norm="reconformer",
            kspace_us=kspace_us,
        )

        zero_fill = centered_ifft2(kspace_us)
        scale = zero_fill.abs().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        self.assertTrue(torch.allclose(model_input, zero_fill / scale, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(target["complex_image"], image / scale, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(target["image"], image.abs(), atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(dc_input, kspace_us, atol=1e-6, rtol=1e-6))
        restored = model_output_to_raw_kspace(model_input, stats, "complex_image")
        self.assertTrue(torch.allclose(restored, kspace_us, atol=1e-6, rtol=1e-6))

    def test_reconformer_loss_matches_source_magnitude_l1(self):
        prediction = torch.tensor([[[[1.0 + 2.0j, 2.0 + 0.0j]]]])
        target = torch.tensor([[[[2.0 + 1.0j, 0.5 + 0.5j]]]])
        criterion = build_loss("reconformer_l1")

        actual = criterion(prediction, {"complex_image": target})
        expected = F.l1_loss(prediction.abs(), target.abs())

        self.assertIsInstance(criterion, ReconFormerMagnitudeL1Loss)
        self.assertTrue(torch.allclose(actual, expected))

    def test_small_model_forward_preserves_sampled_kspace(self):
        torch.manual_seed(0)
        model = ReconFormerBaseline(
            num_ch=(12, 12, 12),
            num_iter=2,
            down_scales=(2.0, 1.0, 1.5),
            img_size=32,
            num_heads=(3, 3, 3),
            depths=(1, 1, 1),
            window_sizes=(8, 8, 8),
            use_checkpoint=(False, False, False, False, False, False),
        )
        full_kspace = torch.randn(1, 1, 32, 32, dtype=torch.complex64)
        mask = torch.zeros(1, 1, 1, 32)
        mask[..., ::4] = 1
        measured = full_kspace * mask
        model_input = centered_ifft2(measured)

        output, intermediates = model(
            model_input,
            measured,
            mask,
            return_intermediates=True,
        )
        output_kspace = reconformer_fft2(output[:, 0]).unsqueeze(1)

        self.assertEqual(output.shape, model_input.shape)
        self.assertEqual(intermediates, [])
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.allclose(output_kspace * mask, measured, atol=2e-5, rtol=2e-5))

    def test_training_epoch_runs_with_current_mask_generator(self):
        cfg = build_cfg(1)
        cfg.image_size = (32, 32)
        cfg.reconformer_num_ch = (12, 12, 12)
        cfg.reconformer_num_iter = 2
        cfg.reconformer_num_heads = (3, 3, 3)
        cfg.reconformer_depths = (1, 1, 1)
        cfg.reconformer_window_sizes = (8, 8, 8)
        model = build_model(cfg)
        optimizer = _build_optimizer(cfg, model.parameters())
        criterion = build_loss(cfg.final_loss_type)
        mask_generator = FastMRIMaskGenerator(
            cfg.acceleration_factors,
            center_fractions=cfg.center_fractions,
            mask_type=cfg.mask_type,
        )
        loader = [torch.randn(1, 1, 32, 32, dtype=torch.complex64)]

        metrics = train_one_epoch(
            cfg,
            model,
            loader,
            cfg.acceleration_factors,
            mask_generator,
            optimizer,
            criterion,
            criterion,
            cfg.loss_mode,
            torch.device("cpu"),
            epoch=0,
        )

        self.assertTrue(np.isfinite(metrics["total_loss"]))
        self.assertTrue(np.isfinite(metrics["psnr"]))
        self.assertEqual(metrics["stage_losses"], [])

    def test_volume_psnr_uses_volume_peak_and_global_mse(self):
        store = {}
        target = torch.tensor([
            [[[1.0, 2.0]]],
            [[[3.0, 4.0]]],
            [[[2.0, 8.0]]],
        ])
        prediction = torch.tensor([
            [[[0.0, 2.0]]],
            [[[2.0, 2.0]]],
            [[[1.0, 6.0]]],
        ])
        _update_volume_errors(store, ["a.h5", "a.h5", "b.h5"], prediction, target)

        mse_a = ((prediction[:2] - target[:2]) ** 2).mean().item()
        mse_b = ((prediction[2:] - target[2:]) ** 2).mean().item()
        expected = np.mean([
            20 * np.log10(4.0) - 10 * np.log10(mse_a),
            20 * np.log10(8.0) - 10 * np.log10(mse_b),
        ])

        self.assertAlmostEqual(_mean_volume_psnr(store), expected, places=6)

    def test_dataset_metadata_preserves_filename_and_h5_max(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "volume.h5"
            kspace = np.zeros((1, 4, 4), dtype=np.complex64)
            with h5py.File(path, "w") as handle:
                handle.create_dataset("kspace", data=kspace)
                handle.attrs["max"] = 7.5

            sample = H5MRIDataset(
                tmpdir,
                image_size=(4, 4),
                return_metadata=True,
            )[0]

            self.assertEqual(sample["fname"], "volume.h5")
            self.assertEqual(sample["slice_num"], 0)
            self.assertEqual(sample["max_value"], 7.5)
            self.assertEqual(sample["kspace"].shape, (1, 4, 4))

    def test_saved_reconformer_config_and_checkpoint_reload(self):
        cfg = build_cfg(1)
        model = build_model(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            experiment = Path(tmpdir)
            with (experiment / "config.json").open("w") as handle:
                json.dump(vars(cfg), handle)
            torch.save({"model": model.state_dict(), "epoch": 0}, experiment / "best_model.pth")

            loaded = load_experiment_model(str(experiment), device=torch.device("cpu"))

            self.assertIsInstance(loaded["model"], ReconFormerBaseline)
            self.assertEqual(loaded["cfg"].model_type, "reconformer")
            self.assertEqual(loaded["checkpoint"]["epoch"], 0)


if __name__ == "__main__":
    unittest.main()
