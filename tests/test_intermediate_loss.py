import sys
import types
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

wandb_stub = types.SimpleNamespace(init=lambda *a, **k: None, log=lambda *a, **k: None, finish=lambda *a, **k: None)
sys.modules.setdefault("wandb", wandb_stub)

from dataset import H5MRIDataset, OASISDataset, centered_fft2, centered_ifft2, center_crop_complex
from DcTNN.loss import (
    ComplexL1Loss,
    ComplexL2Loss,
    MagnitudeImageLoss,
    MagnitudeL1Loss,
    PerpendicularLoss,
    _perpendicular_mag_weight_map,
    build_loss,
)
from DcTNN.lambda_scheduler import LambdaScheduler
from DcTNN.model import cascadeNet
from inference import _denormalize_image, _flat_to_cfg, to_image_magnitude
from normalizer import (
    apply_kspace_companding,
    apply_log_kspace,
    invert_kspace_companding,
    invert_log_kspace,
    invert_normalization,
    kspace_to_image_magnitude,
    model_output_to_raw_kspace,
    restore_original_kspace,
    robust_shifted,
)
from train import (
    _build_lambda_scheduler,
    _compute_losses,
    _find_executable_batch_size,
    _probe_batch_candidate,
    _psnr_per_sample,
    _resolve_batch_size,
    build_cfg,
    train_one_epoch,
    validate,
)
from train_config import EXPERIMENTS
from train_utils import build_model, simulate_undersampling


class _DummyEncoder(nn.Module):
    def __init__(self, image_size, delta=0.0, **kwargs):
        super().__init__()
        self.delta = delta

    def forward(self, x, col_mask=None):
        return torch.full_like(x, self.delta)


class _MetricModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.transformers = nn.ModuleList([nn.Identity()])

    def forward(self, x, y, mask, return_intermediates=False, stats=None):
        recon = x + self.bias
        return (recon, [recon]) if return_intermediates else recon


class _MetricMaskGenerator:
    def __init__(self, *args, **kwargs):
        pass

    def apply(self, kspace_full, acceleration, seed=None):
        mask = torch.ones(1, 1, 1, kspace_full.shape[-1], device=kspace_full.device)
        return kspace_full, mask, kspace_full.shape[-1]


def _metric_simulation(kspace_full, *args, **kwargs):
    if kspace_full.shape[0] == 2:
        model_input = torch.tensor([0.0, 5.0], device=kspace_full.device).reshape(2, 1, 1, 1)
        target = torch.tensor([1.0, 10.0], device=kspace_full.device).reshape(2, 1, 1, 1)
    else:
        model_input = torch.zeros(1, 1, 1, 1, device=kspace_full.device)
        target = torch.full((1, 1, 1, 1), 2.0, device=kspace_full.device)
    return model_input, kspace_full, {"image": target}, None


def _metric_cfg():
    return types.SimpleNamespace(
        seed=42,
        grad_clip=1.0,
        center_fractions=None,
        mask_type="random",
        learning="k_space",
        norm="none",
        robust_clip=3.0,
        robust_shift=3.0,
        companding_p=0.8,
        companding_a=0.5,
        companding_centering="fft",
    )


def _batch_probe_cfg():
    cfg = _metric_cfg()
    cfg.acceleration_factors = [4]
    cfg.lr = 1e-3
    cfg.weight_decay = 0.0
    cfg.final_loss_type = "l1"
    cfg.intermediate_loss_type = "l1"
    cfg.perpendicular_mag_weighting = False
    cfg.loss_mode = "final_only"
    cfg.batch_size_probe_steps = 3
    return cfg


class IntermediateLossTest(unittest.TestCase):
    def test_fastmri_dataset_crops_in_image_domain_to_requested_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kspace = np.zeros((1, 4, 6), dtype=np.complex64)
            kspace[0, 2, 3] = 1.0 + 0.0j
            fpath = Path(tmpdir) / "sample.h5"
            with h5py.File(fpath, "w") as f:
                f.create_dataset("kspace", data=kspace)

            ds = H5MRIDataset(tmpdir, image_size=(2, 2))
            sample = ds[0]

            raw = torch.tensor(kspace[0], dtype=torch.complex64)
            image = centered_ifft2(raw)
            expected = centered_fft2(center_crop_complex(image, 2, 2)).unsqueeze(0)

            self.assertEqual(tuple(sample.shape), (1, 2, 2))
            self.assertTrue(sample.is_complex())
            self.assertTrue(torch.allclose(sample, expected, atol=1e-6, rtol=1e-6))

    def test_fastmri_dataset_max_val_files_caps_files_not_slices(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            total_slices = 0
            for idx in range(7):
                slices = idx + 1
                total_slices += slices
                fpath = Path(tmpdir) / f"sample_{idx}.h5"
                with h5py.File(fpath, "w") as f:
                    f.create_dataset("kspace", data=np.zeros((slices, 4, 4), dtype=np.complex64))

            ds = H5MRIDataset(tmpdir, max_files=5)
            self.assertEqual(len(ds.h5_files), 5)
            self.assertEqual(len(ds), sum(range(1, 6)))

    def test_build_loss_supports_l1_and_l2(self):
        self.assertIsInstance(build_loss("l1"), MagnitudeL1Loss)
        self.assertIsInstance(build_loss("l2"), MagnitudeImageLoss)
        self.assertIsInstance(build_loss("image_domain_l1"), MagnitudeL1Loss)
        self.assertIsInstance(build_loss("image_domain_l2"), MagnitudeImageLoss)
        self.assertIsInstance(build_loss("complex_l1"), ComplexL1Loss)
        self.assertIsInstance(build_loss("complex_l2"), ComplexL2Loss)
        self.assertIsInstance(build_loss("perpendicular_loss"), PerpendicularLoss)
        weighted_loss = build_loss("perpendicular_loss", magnitude_weighting=True, magnitude_weight_m=2.0)
        self.assertTrue(weighted_loss.magnitude_weighting)
        self.assertAlmostEqual(weighted_loss.current_m, 2.0, places=6)
        with self.assertRaisesRegex(ValueError, "Unknown loss_type"):
            build_loss("bad")

    def test_compute_losses_final_only_logs_intermediates_without_optimizing_them(self):
        gt = {"image": torch.ones(1, 1, 2, 2), "kspace": torch.ones(1, 1, 2, 2, dtype=torch.complex64)}
        recon = torch.full((1, 1, 2, 2), 2.0)
        intermediates = [torch.full((1, 1, 2, 2), 3.0), torch.full((1, 1, 2, 2), 4.0)]

        total_loss, final_loss, intermediate_loss_sum, stage_losses, stage_psnr_gains = _compute_losses(
            recon, intermediates, gt, build_loss("l1"), build_loss("l2"), "final_only"
        )

        self.assertAlmostEqual(final_loss.item(), 1.0, places=6)
        self.assertAlmostEqual(stage_losses[0].item(), 4.0, places=6)
        self.assertAlmostEqual(stage_losses[1].item(), 9.0, places=6)
        self.assertAlmostEqual(intermediate_loss_sum.item(), 13.0, places=6)
        self.assertAlmostEqual(total_loss.item(), final_loss.item(), places=6)
        self.assertEqual(stage_psnr_gains, [])

    def test_compute_losses_intermediate_unweighted_sums_stage_losses(self):
        gt = {"image": torch.zeros(1, 1, 2, 2), "kspace": torch.zeros(1, 1, 2, 2, dtype=torch.complex64)}
        recon = torch.ones(1, 1, 2, 2)
        intermediates = [torch.ones(1, 1, 2, 2), torch.full((1, 1, 2, 2), 2.0)]

        total_loss, final_loss, intermediate_loss_sum, stage_losses, stage_psnr_gains = _compute_losses(
            recon, intermediates, gt, build_loss("l2"), build_loss("l1"), "intermediate_unweighted"
        )

        self.assertAlmostEqual(final_loss.item(), 1.0, places=6)
        self.assertAlmostEqual(stage_losses[0].item(), 1.0, places=6)
        self.assertAlmostEqual(stage_losses[1].item(), 2.0, places=6)
        self.assertAlmostEqual(intermediate_loss_sum.item(), 3.0, places=6)
        self.assertAlmostEqual(total_loss.item(), 4.0, places=6)
        self.assertEqual(stage_psnr_gains, [])

    def test_compute_losses_reports_stage_psnr_gains_in_image_domain(self):
        gt_img = torch.ones(1, 1, 2, 2)
        target = {"image": gt_img, "kspace": torch.zeros(1, 1, 2, 2, dtype=torch.complex64)}
        zf = torch.zeros(1, 1, 2, 2)
        stage1 = torch.full((1, 1, 2, 2), 0.5)
        stage2 = torch.full((1, 1, 2, 2), 0.75)
        recon = torch.full((1, 1, 2, 2), 0.9)

        _, _, _, _, stage_psnr_gains = _compute_losses(
            recon, [stage1, stage2], target, build_loss("image_domain_l1"), build_loss("image_domain_l1"),
            "intermediate_unweighted", zf_recon=zf
        )

        self.assertEqual(len(stage_psnr_gains), 2)
        expected0 = 20.0 * torch.log10(gt_img.max() / torch.sqrt(torch.mean((stage1 - gt_img) ** 2)))
        expected_zf = 20.0 * torch.log10(gt_img.max() / torch.sqrt(torch.mean((zf - gt_img) ** 2)))
        expected1 = 20.0 * torch.log10(gt_img.max() / torch.sqrt(torch.mean((stage2 - gt_img) ** 2)))
        self.assertAlmostEqual(stage_psnr_gains[0].item(), (expected0 - expected_zf).item(), places=6)
        self.assertAlmostEqual(stage_psnr_gains[1].item(), (expected1 - expected0).item(), places=6)

    def test_psnr_per_sample_uses_each_sample_peak_and_error(self):
        target = torch.stack((torch.ones(1, 2, 2), torch.full((1, 2, 2), 10.0)))
        pred = torch.stack((torch.zeros(1, 2, 2), torch.full((1, 2, 2), 5.0)))

        actual = _psnr_per_sample(pred, target)
        expected = torch.tensor([0.0, 20.0 * torch.log10(torch.tensor(2.0))])

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))
        self.assertAlmostEqual(actual.mean().item(), 3.0103, places=4)

    def test_train_metrics_are_averaged_per_sample(self):
        cfg = _metric_cfg()
        model = _MetricModel()
        loader = [
            torch.zeros(2, 1, 1, 1, dtype=torch.cfloat),
            torch.zeros(1, 1, 1, 1, dtype=torch.cfloat),
        ]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        criterion = build_loss("l1")

        with patch("train.simulate_undersampling", side_effect=_metric_simulation):
            metrics = train_one_epoch(
                cfg, model, loader, [4], _MetricMaskGenerator(), optimizer,
                criterion, criterion, "final_only", torch.device("cpu"), 0,
            )

        self.assertAlmostEqual(metrics["total_loss"], 8.0 / 3.0, places=6)
        self.assertAlmostEqual(metrics["final_loss"], 8.0 / 3.0, places=6)
        self.assertAlmostEqual(metrics["intermediate_loss_sum"], 8.0 / 3.0, places=6)
        self.assertAlmostEqual(metrics["stage_losses"][0], 8.0 / 3.0, places=6)
        self.assertAlmostEqual(metrics["psnr"], 20.0 * torch.log10(torch.tensor(2.0)).item() / 3.0, places=6)

    def test_validation_losses_are_averaged_per_sample(self):
        cfg = _metric_cfg()
        model = _MetricModel()
        loader = [
            torch.zeros(2, 1, 1, 1, dtype=torch.cfloat),
            torch.zeros(1, 1, 1, 1, dtype=torch.cfloat),
        ]
        criterion = build_loss("l1")

        with patch("train.FastMRIMaskGenerator", _MetricMaskGenerator), patch(
            "train.simulate_undersampling", side_effect=_metric_simulation
        ):
            metrics = validate(
                cfg, model, loader, [4], (1, 1), criterion, criterion,
                "final_only", torch.device("cpu"),
            )

        self.assertAlmostEqual(metrics["total_loss"], 8.0 / 3.0, places=6)
        self.assertAlmostEqual(metrics["final_loss"], 8.0 / 3.0, places=6)
        self.assertAlmostEqual(metrics["intermediate_loss_sum"], 8.0 / 3.0, places=6)
        self.assertAlmostEqual(metrics["stage_losses"][0], 8.0 / 3.0, places=6)
        self.assertAlmostEqual(metrics["psnr"], 20.0 * torch.log10(torch.tensor(2.0)).item() / 3.0, places=6)

    def test_batch_finder_halves_until_probe_passes(self):
        attempted = []

        def probe(batch_size):
            attempted.append(batch_size)
            if batch_size > 32:
                raise torch.OutOfMemoryError("CUDA out of memory")

        selected = _find_executable_batch_size(128, probe)

        self.assertEqual(selected, 32)
        self.assertEqual(attempted, [128, 64, 32])

    def test_batch_finder_reraises_non_oom_runtime_errors(self):
        def probe(_batch_size):
            raise RuntimeError("shape mismatch")

        with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
            _find_executable_batch_size(128, probe)

    def test_batch_finder_fails_when_batch_size_one_ooms(self):
        def probe(_batch_size):
            raise torch.OutOfMemoryError("CUDA out of memory")

        with self.assertRaisesRegex(RuntimeError, "cannot fit a batch size of 1"):
            _find_executable_batch_size(1, probe)

    def test_batch_probe_runs_three_real_optimizer_steps(self):
        cfg = _batch_probe_cfg()
        model = _MetricModel()
        dataset = [torch.zeros(1, 1, 1, dtype=torch.cfloat) for _ in range(4)]

        with patch("train.build_model", return_value=model), patch(
            "train.FastMRIMaskGenerator", _MetricMaskGenerator
        ), patch("train.simulate_undersampling", side_effect=_metric_simulation) as simulation:
            _probe_batch_candidate(cfg, dataset, 2, torch.device("cpu"))

        self.assertEqual(simulation.call_count, 3)

    def test_automatic_batch_search_is_skipped_without_cuda(self):
        cfg = types.SimpleNamespace(auto_batch_size=True, batch_size=32)

        selected = _resolve_batch_size(cfg, [torch.zeros(1)], torch.device("cpu"))

        self.assertEqual(selected, 32)

    def test_kspace_companding_forward_inverse_recovers_input_and_preserves_phase(self):
        real = torch.tensor([[[[1.0, -2.0], [0.5, -0.25]]]])
        imag = torch.tensor([[[[0.25, 1.5], [-0.75, 2.0]]]])
        ks = torch.complex(real, imag)

        companded = apply_kspace_companding(ks, a=0.5, p=0.8)
        restored = invert_kspace_companding(companded, a=0.5, p=0.8)

        self.assertTrue(torch.allclose(restored, ks, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(torch.angle(companded), torch.angle(ks), atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.isfinite(companded.abs()).all())

    def test_kspace_companding_centers_unique_minimum_on_shifted_fft_origin(self):
        ks = torch.ones(1, 1, 4, 6, dtype=torch.complex64)

        companded = apply_kspace_companding(ks, a=0.5, p=0.8)
        minimum = companded.abs().amin()
        minimum_indices = torch.nonzero(companded.abs() == minimum)

        self.assertEqual(minimum_indices.tolist(), [[0, 0, 2, 3]])
        self.assertAlmostEqual(companded.abs()[0, 0, 2, 3].item(), 1e-6, places=10)

    def test_legacy_companding_stats_restore_legacy_centering(self):
        ks = torch.randn(1, 1, 4, 6, dtype=torch.complex64)
        companded = apply_kspace_companding(ks, a=0.5, p=0.8, centering="legacy")
        stats = {"normalization": "kspace_companding", "companding_p": 0.8, "companding_a": 0.5}

        restored = restore_original_kspace(companded, stats)

        self.assertTrue(torch.allclose(restored, ks, atol=1e-5, rtol=1e-5))
        self.assertEqual(_flat_to_cfg({"norm": "kspace_companding"}).companding_centering, "legacy")
        self.assertEqual(
            _flat_to_cfg({"norm": "kspace_companding", "companding_centering": "fft"}).companding_centering,
            "fft",
        )

    def test_log_kspace_forward_inverse_recovers_input_and_preserves_phase(self):
        real = torch.tensor([[[[0.0, -2.0], [0.5, -0.25]]]])
        imag = torch.tensor([[[[0.0, 1.5], [-0.75, 2.0]]]])
        ks = torch.complex(real, imag)

        logged = apply_log_kspace(ks)
        restored = invert_log_kspace(logged)

        self.assertTrue(torch.allclose(restored, ks, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(torch.angle(logged), torch.angle(ks), atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.isfinite(logged.abs()).all())

    def test_simulate_undersampling_kspace_companding_returns_companded_kspace_and_targets(self):
        img = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]], dtype=torch.float32)
        kspace_full = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(img), norm='ortho')).to(torch.complex64)
        mask = torch.ones_like(kspace_full.real)

        model_input, dc_input, target, stats = simulate_undersampling(
            kspace_full,
            mask,
            learning="k_space",
            norm="kspace_companding",
            companding_p=0.8,
            companding_a=0.5,
        )

        self.assertTrue(model_input.is_complex())
        self.assertTrue(dc_input.is_complex())
        self.assertEqual(stats["normalization"], "kspace_companding")
        self.assertEqual(stats["companding_centering"], "fft")
        self.assertTrue(torch.allclose(model_input, apply_kspace_companding(kspace_full, a=0.5, p=0.8), atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(dc_input, kspace_full, atol=1e-6, rtol=1e-6))
        expected_image_gt = kspace_to_image_magnitude(apply_kspace_companding(kspace_full, a=0.5, p=0.8), stats)
        self.assertTrue(torch.allclose(target["image"], expected_image_gt, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(target["kspace"], apply_kspace_companding(kspace_full, a=0.5, p=0.8), atol=1e-6, rtol=1e-6))

    def test_simulate_undersampling_kspace_companding_rejects_image_learning(self):
        kspace_full = torch.ones(1, 1, 2, 2, dtype=torch.complex64)
        mask = torch.ones(1, 1, 2, 2)

        with self.assertRaisesRegex(ValueError, "only supported when learning='k_space'"):
            simulate_undersampling(kspace_full, mask, learning="image", norm="kspace_companding")

    def test_simulate_undersampling_log_kspace_returns_logged_kspace_and_targets(self):
        img = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]], dtype=torch.float32)
        kspace_full = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(img), norm='ortho')).to(torch.complex64)
        mask = torch.ones_like(kspace_full.real)

        model_input, dc_input, target, stats = simulate_undersampling(
            kspace_full,
            mask,
            learning="k_space",
            norm="log_kspace",
        )

        self.assertTrue(model_input.is_complex())
        self.assertTrue(dc_input.is_complex())
        self.assertEqual(stats["normalization"], "log_kspace")
        self.assertTrue(torch.allclose(model_input, apply_log_kspace(kspace_full), atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(dc_input, kspace_full, atol=1e-6, rtol=1e-6))
        expected_image_gt = kspace_to_image_magnitude(apply_log_kspace(kspace_full), stats)
        self.assertTrue(torch.allclose(target["image"], expected_image_gt, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(target["kspace"], apply_log_kspace(kspace_full), atol=1e-6, rtol=1e-6))

    def test_simulate_undersampling_log_kspace_rejects_image_learning(self):
        kspace_full = torch.ones(1, 1, 2, 2, dtype=torch.complex64)
        mask = torch.ones(1, 1, 2, 2)

        with self.assertRaisesRegex(ValueError, "only supported when learning='k_space'"):
            simulate_undersampling(kspace_full, mask, learning="image", norm="log_kspace")

    def test_companded_l1_loss_uses_normalized_kspace_magnitude(self):
        img = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)
        pred_img = img * 1.1
        gt_kspace = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(img), norm='ortho')).to(torch.complex64)
        pred_kspace = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(pred_img), norm='ortho')).to(torch.complex64)
        stats = {"normalization": "kspace_companding", "companding_p": 0.8, "companding_a": 0.5, "companding_centering": "fft"}

        gt_comp = apply_kspace_companding(gt_kspace, a=0.5, p=0.8)
        pred_comp = apply_kspace_companding(pred_kspace, a=0.5, p=0.8)

        loss = build_loss("l1")
        companded_loss = loss(pred_comp, {"image": img, "kspace": gt_comp}, stats=stats)
        expected = torch.mean(torch.abs(pred_comp.abs() - gt_comp.abs()))

        self.assertAlmostEqual(companded_loss.item(), expected.item(), places=5)

    def test_log_kspace_l2_loss_uses_normalized_kspace_magnitude(self):
        gt = torch.tensor([[[[1 + 2j, 0.5 + 0.25j]]]], dtype=torch.complex64)
        pred = torch.tensor([[[[1.2 + 1.8j, 0.25 + 0.4j]]]], dtype=torch.complex64)
        gt_log = apply_log_kspace(gt)
        pred_log = apply_log_kspace(pred)
        stats = {"normalization": "log_kspace"}

        loss = build_loss("l2")
        actual = loss(pred_log, {"image": torch.zeros(1, 1, 1, 2), "kspace": gt_log}, stats=stats)
        expected = torch.mean((pred_log.abs() - gt_log.abs()) ** 2)

        self.assertAlmostEqual(actual.item(), expected.item(), places=6)

    def test_log_kspace_complex_losses_run_against_normalized_kspace(self):
        gt = torch.tensor([[[[1 + 2j, 0.5 + 0.25j]]]], dtype=torch.complex64)
        pred = torch.tensor([[[[1.2 + 1.8j, 0.25 + 0.4j]]]], dtype=torch.complex64)
        gt_log = apply_log_kspace(gt)
        pred_log = apply_log_kspace(pred)
        target = {"image": torch.zeros(1, 1, 1, 2), "kspace": gt_log}

        complex_l1 = build_loss("complex_l1")(pred_log, target, stats={"normalization": "log_kspace"})
        expected = torch.mean(torch.abs(pred_log.real - gt_log.real) + torch.abs(pred_log.imag - gt_log.imag))
        self.assertAlmostEqual(complex_l1.item(), expected.item(), places=6)

    def test_restore_original_kspace_inverts_supported_normalizations(self):
        ks = torch.tensor([[[[1 + 2j, 0.5 + 0.25j]]]], dtype=torch.complex64)
        comp_stats = {"normalization": "kspace_companding", "companding_p": 0.8, "companding_a": 0.5, "companding_centering": "fft"}
        log_stats = {"normalization": "log_kspace"}

        self.assertTrue(torch.allclose(restore_original_kspace(apply_kspace_companding(ks, 0.5, 0.8), comp_stats), ks, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(restore_original_kspace(apply_log_kspace(ks), log_stats), ks, atol=1e-5, rtol=1e-5))

    def test_restore_original_kspace_inverts_fastmri_magnitude_per_batch(self):
        ks = torch.tensor([
            [[[1 + 2j, 0.5 + 0.25j]]],
            [[[3 + 4j, 1.5 + 0.75j]]],
        ], dtype=torch.complex64)
        p95 = torch.tensor([2.0, 4.0])
        normalized = ks / p95.reshape(-1, 1, 1, 1)

        restored = restore_original_kspace(
            normalized,
            {"normalization": "fastmri_magnitude", "p95": p95},
        )
        restored_single = restore_original_kspace(
            normalized[:1],
            {"normalization": "fastmri_magnitude", "p95": p95[0]},
        )

        self.assertTrue(torch.allclose(restored, ks, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(restored_single, ks[:1], atol=1e-6, rtol=1e-6))

    def test_fastmri_magnitude_restored_image_matches_raw_target(self):
        image = torch.tensor([
            [[[1.0, 2.0], [3.0, 4.0]]],
            [[[2.0, 1.0], [4.0, 3.0]]],
        ])
        kspace = centered_fft2(image)
        mask = torch.ones_like(kspace.real)

        model_input, _, target, stats = simulate_undersampling(
            kspace,
            mask,
            learning="k_space",
            norm="fastmri_magnitude",
        )

        self.assertTrue(torch.allclose(restore_original_kspace(model_input, stats), kspace, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(kspace_to_image_magnitude(model_input, stats), image, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(target["image"], image, atol=1e-5, rtol=1e-5))

    def test_fastmri_kspace_mode_normalizes_directly_by_kspace_p95(self):
        torch.manual_seed(7)
        image = torch.randn(2, 1, 8, 8, dtype=torch.complex64)
        kspace = centered_fft2(image)
        mask = torch.zeros_like(kspace.real)
        mask[..., ::4] = 1
        kspace_us = kspace * mask

        model_input, dc_input, target, stats = simulate_undersampling(
            kspace,
            mask,
            learning="k_space",
            norm="fastmri_magnitude",
            kspace_us=kspace_us,
        )

        scale = torch.quantile(
            kspace_us.abs().reshape(kspace_us.shape[0], -1), q=0.95, dim=1
        ).clamp_min(1e-8).reshape(-1, 1, 1, 1)
        expected_input = kspace_us / scale
        expected_target = kspace / scale
        self.assertEqual(stats["normalization_domain"], "k_space")
        self.assertTrue(torch.allclose(stats["p95"], scale))
        self.assertTrue(torch.allclose(model_input, expected_input))
        self.assertTrue(torch.allclose(dc_input, kspace_us))
        self.assertTrue(torch.allclose(target["kspace"], expected_target))
        self.assertTrue(torch.allclose(model_input * mask, target["kspace"] * mask))
        self.assertEqual(torch.count_nonzero(model_input * (1 - mask)).item(), 0)
        self.assertTrue(torch.allclose(
            torch.quantile(model_input.abs().reshape(2, -1), q=0.95, dim=1),
            torch.ones(2),
            atol=1e-5,
            rtol=1e-5,
        ))
        self.assertTrue(torch.allclose(
            restore_original_kspace(model_input, stats),
            kspace_us,
            atol=1e-5,
            rtol=1e-5,
        ))
        self.assertTrue(torch.allclose(
            target["image"], image.abs(), atol=1e-5, rtol=1e-5
        ))

    def test_fastmri_complex_image_mode_preserves_phase_and_uses_complex_image_target(self):
        image = torch.tensor([[[[1 + 2j, 2 - 1j], [3 + 0.5j, 4 - 2j]]]], dtype=torch.complex64)
        kspace = centered_fft2(image)
        mask = torch.ones_like(kspace.real)

        model_input, dc_input, target, stats = simulate_undersampling(
            kspace,
            mask,
            learning="complex_image",
            norm="fastmri_magnitude",
        )

        scale = torch.quantile(image.abs().reshape(1, -1), q=0.95, dim=1).reshape(1, 1, 1, 1)
        expected_image = image / scale
        self.assertEqual(stats["prediction_domain"], "complex_image")
        self.assertEqual(stats["normalization_domain"], "complex_image")
        self.assertTrue(model_input.is_complex())
        self.assertTrue(torch.allclose(model_input, expected_image, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(dc_input, kspace, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(target["complex_image"], expected_image, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(target["image"], image.abs(), atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(to_image_magnitude(model_input, stats), image.abs(), atol=1e-5, rtol=1e-5))
        for loss_name in ("complex_l1", "complex_l2", "perpendicular_loss"):
            self.assertAlmostEqual(build_loss(loss_name)(model_input, target, stats=stats).item(), 0.0, places=6)

    def test_robust_shifted_normalization_round_trips_in_each_complex_domain(self):
        torch.manual_seed(11)
        image = torch.randn(2, 1, 8, 8, dtype=torch.complex64)
        kspace = centered_fft2(image)
        mask = torch.zeros_like(kspace.real)
        mask[..., ::4] = 1
        kspace_us = kspace * mask

        for learning in ("k_space", "complex_image"):
            with self.subTest(learning=learning):
                model_input, dc_input, target, stats = simulate_undersampling(
                    kspace,
                    mask,
                    learning=learning,
                    norm="robust_shifted",
                    kspace_us=kspace_us,
                    robust_clip=3.0,
                    robust_shift=3.0,
                )
                target_tensor = (
                    target["kspace"]
                    if learning == "k_space"
                    else target["complex_image"]
                )
                raw_target_kspace = model_output_to_raw_kspace(
                    target_tensor, stats, learning
                )

                self.assertEqual(stats["normalization_domain"], learning)
                self.assertTrue(torch.allclose(dc_input, kspace_us))
                self.assertTrue(torch.allclose(
                    raw_target_kspace, kspace, atol=2e-3, rtol=2e-3
                ))
                self.assertTrue(torch.allclose(
                    target["image"], image.abs(), atol=1e-5, rtol=1e-5
                ))
                self.assertTrue(torch.allclose(
                    to_image_magnitude(target_tensor, stats),
                    image.abs(),
                    atol=2e-3,
                    rtol=2e-3,
                ))
                expected_denormalized = kspace if learning == "k_space" else image
                self.assertTrue(torch.allclose(
                    _denormalize_image(target_tensor, stats),
                    expected_denormalized,
                    atol=2e-3,
                    rtol=2e-3,
                ))
                if learning == "k_space":
                    self.assertTrue(torch.allclose(
                        model_input * mask, target_tensor * mask,
                        atol=1e-6, rtol=1e-6,
                    ))
                    self.assertEqual(
                        torch.count_nonzero(model_input * (1 - mask)).item(), 0
                    )
                    measured_magnitudes = model_input[mask.bool()]
                    self.assertGreaterEqual(measured_magnitudes.abs().min().item(), 0.0)
                    self.assertLessEqual(measured_magnitudes.abs().max().item(), 6.0)
                else:
                    self.assertGreaterEqual(model_input.abs().min().item(), 0.0)
                    self.assertLessEqual(model_input.abs().max().item(), 6.0)

    def test_normalized_models_apply_data_consistency_in_raw_kspace(self):
        torch.manual_seed(13)
        image = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
        kspace = centered_fft2(image)
        mask = torch.zeros_like(kspace.real)
        mask[..., ::4] = 1
        kspace_us = kspace * mask

        for learning in ("k_space", "complex_image"):
            with self.subTest(learning=learning):
                model_input, dc_input, _, stats = simulate_undersampling(
                    kspace,
                    mask,
                    learning=learning,
                    norm="robust_shifted",
                    kspace_us=kspace_us,
                )
                model = cascadeNet(
                    N=(8, 8),
                    encList=[_DummyEncoder],
                    encArgs=[{"delta": 0.25}],
                    lamb=False,
                    learning=learning,
                )
                reconstruction = model(
                    model_input, dc_input, mask, stats=stats
                )
                raw_reconstruction = model_output_to_raw_kspace(
                    reconstruction, stats, learning
                )

                self.assertTrue(torch.allclose(
                    raw_reconstruction * mask,
                    kspace_us * mask,
                    atol=2e-4,
                    rtol=2e-4,
                ))

    def test_all_supported_normalizers_apply_dc_in_raw_kspace(self):
        torch.manual_seed(17)
        image = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
        kspace = centered_fft2(image)
        mask = torch.zeros_like(kspace.real)
        mask[..., ::4] = 1
        kspace_us = kspace * mask
        cases = [
            ("none", "k_space"),
            ("none", "complex_image"),
            ("zscore", "k_space"),
            ("zscore", "complex_image"),
            ("fastmri_magnitude", "k_space"),
            ("fastmri_magnitude", "complex_image"),
            ("robust_shifted", "k_space"),
            ("robust_shifted", "complex_image"),
            ("kspace_companding", "k_space"),
            ("log_kspace", "k_space"),
        ]

        for normalization, learning in cases:
            with self.subTest(normalization=normalization, learning=learning):
                model_input, dc_input, _, stats = simulate_undersampling(
                    kspace,
                    mask,
                    learning=learning,
                    norm=normalization,
                    kspace_us=kspace_us,
                )
                model = cascadeNet(
                    N=(8, 8),
                    encList=[_DummyEncoder],
                    encArgs=[{"delta": 0.1}],
                    lamb=False,
                    learning=learning,
                )
                reconstruction = model(
                    model_input, dc_input, mask, stats=stats
                )
                raw_reconstruction = model_output_to_raw_kspace(
                    reconstruction, stats, learning
                )

                self.assertTrue(torch.allclose(dc_input, kspace_us))
                self.assertTrue(torch.allclose(
                    raw_reconstruction * mask,
                    kspace_us,
                    atol=2e-3,
                    rtol=2e-3,
                ))

    def test_inference_denormalize_returns_original_kspace_for_supported_kspace_norms(self):
        ks = torch.tensor([[[[1 + 2j, 0.5 + 0.25j]]]], dtype=torch.complex64)
        comp_stats = {"normalization": "kspace_companding", "companding_p": 0.8, "companding_a": 0.5, "companding_centering": "fft"}
        log_stats = {"normalization": "log_kspace"}
        fastmri_stats = {"normalization": "fastmri_magnitude", "p95": torch.tensor(2.0)}

        comp = apply_kspace_companding(ks, a=0.5, p=0.8)
        logged = apply_log_kspace(ks)
        fastmri_normalized = ks / fastmri_stats["p95"]

        self.assertTrue(torch.allclose(_denormalize_image(comp, comp_stats), ks, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(_denormalize_image(logged, log_stats), ks, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(_denormalize_image(fastmri_normalized, fastmri_stats), ks, atol=1e-5, rtol=1e-5))

    def test_to_image_magnitude_restores_kspace_before_ifft(self):
        img = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)
        kspace = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(img), norm='ortho')).to(torch.complex64)
        comp_stats = {"normalization": "kspace_companding", "companding_p": 0.8, "companding_a": 0.5, "companding_centering": "fft"}
        log_stats = {"normalization": "log_kspace"}

        comp_mag = to_image_magnitude(apply_kspace_companding(kspace, a=0.5, p=0.8), comp_stats)
        log_mag = to_image_magnitude(apply_log_kspace(kspace), log_stats)

        self.assertTrue(torch.allclose(comp_mag, img, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(log_mag, img, atol=1e-5, rtol=1e-5))

    def test_complex_losses_match_manual_definitions(self):
        pred = torch.tensor([[[[1 + 2j, 3 + 4j]]]], dtype=torch.complex64)
        gt = torch.tensor([[[[0 + 1j, 1 + 1j]]]], dtype=torch.complex64)
        target = {"image": torch.zeros(1, 1, 1, 2), "kspace": gt}

        complex_l1 = build_loss("complex_l1")(pred, target)
        complex_l2 = build_loss("complex_l2")(pred, target)

        manual_l1 = torch.mean(torch.abs(pred.real - gt.real) + torch.abs(pred.imag - gt.imag))
        manual_l2 = torch.mean((pred.real - gt.real) ** 2 + (pred.imag - gt.imag) ** 2)
        self.assertAlmostEqual(complex_l1.item(), manual_l1.item(), places=6)
        self.assertAlmostEqual(complex_l2.item(), manual_l2.item(), places=6)

    def test_perpendicular_loss_accepts_complex_kspace(self):
        pred = torch.tensor([[[[1 + 1j, 0.5 + 2j]]]], dtype=torch.complex64)
        gt = torch.tensor([[[[1 + 0j, 2 + 0.5j]]]], dtype=torch.complex64)
        target = {"image": torch.zeros(1, 1, 1, 2), "kspace": gt}

        loss = build_loss("perpendicular_loss")(pred, target)
        self.assertTrue(torch.isfinite(loss))

    def test_perpendicular_loss_penalizes_quadrature_phase(self):
        target = torch.ones(1, dtype=torch.complex64)
        prediction = torch.tensor([1j], dtype=torch.complex64)

        loss = PerpendicularLoss()(prediction, target)

        self.assertAlmostEqual(loss.item(), 1.0, places=6)

    def test_perpendicular_loss_is_continuous_and_monotonic_across_quadrature(self):
        target = torch.ones(5, dtype=torch.complex64)
        angles = torch.tensor([0.0, torch.pi / 4, torch.pi / 2, 3 * torch.pi / 4, torch.pi])
        predictions = torch.polar(torch.ones_like(angles), angles)
        criterion = PerpendicularLoss()
        losses = torch.stack([
            criterion(prediction, target_value)
            for prediction, target_value in zip(predictions, target)
        ])
        offset = 1e-4
        near_angles = torch.tensor([torch.pi / 2 - offset, torch.pi / 2 + offset])
        near_predictions = torch.polar(torch.ones_like(near_angles), near_angles)
        near_losses = torch.stack([
            criterion(prediction, torch.ones_like(prediction))
            for prediction in near_predictions
        ])

        self.assertTrue(torch.all(losses[1:] > losses[:-1]))
        self.assertLess(torch.abs(near_losses[0] - near_losses[1]).item(), 1e-5)

    def test_perpendicular_loss_weighting_disabled_matches_current_behavior(self):
        pred = torch.tensor([[[[1 + 1j, 0.5 + 2j], [0.25 + 0.5j, 1.5 + 0.25j]]]], dtype=torch.complex64)
        gt = torch.tensor([[[[1 + 0j, 2 + 0.5j], [0.5 + 0.5j, 0.75 + 1.0j]]]], dtype=torch.complex64)
        target = {"image": torch.zeros(1, 1, 2, 2), "kspace": gt}

        base_loss = PerpendicularLoss()(pred, target)
        weighted_off_loss = PerpendicularLoss(magnitude_weighting=False, magnitude_weight_m=3.0)(pred, target)
        self.assertAlmostEqual(base_loss.item(), weighted_off_loss.item(), places=6)

    def test_perpendicular_loss_weighting_with_m_one_is_identity(self):
        pred = torch.tensor([[[[1 + 1j, 0.5 + 2j], [0.25 + 0.5j, 1.5 + 0.25j]]]], dtype=torch.complex64)
        gt = torch.tensor([[[[1 + 0j, 2 + 0.5j], [0.5 + 0.5j, 0.75 + 1.0j]]]], dtype=torch.complex64)
        target = {"image": torch.zeros(1, 1, 2, 2), "kspace": gt}

        base_loss = PerpendicularLoss()(pred, target)
        weighted_identity_loss = PerpendicularLoss(magnitude_weighting=True, magnitude_weight_m=1.0)(pred, target)
        self.assertAlmostEqual(base_loss.item(), weighted_identity_loss.item(), places=6)

    def test_perpendicular_weight_map_has_expected_piecewise_shape(self):
        x = torch.zeros(1, 1, 256, 256, dtype=torch.complex64)
        weights = _perpendicular_mag_weight_map(x, m=3.0, k=0.103, p=67.0)

        expected_center = ((3.0 - 1.0) * torch.exp(torch.tensor(0.103 * (0.0 - 67.0))) + 1.0).item()
        self.assertAlmostEqual(weights[0, 0, 128, 128].item(), expected_center, places=6)
        self.assertAlmostEqual(weights[0, 0, 128, 128 + 67].item(), 1.0, places=6)
        self.assertGreater(weights[0, 0, 128, 128 + 10].item(), 1.0)
        self.assertGreater(weights[0, 0, 128, 128 + 66].item(), 1.0)
        left_near_cutoff = ((3.0 - 1.0) * torch.exp(torch.tensor(0.103 * (66.999 - 67.0))) + 1.0).item()
        self.assertAlmostEqual(left_near_cutoff, 3.0, places=3)

    def test_perpendicular_loss_weighting_only_scales_magnitude_term(self):
        pred = torch.tensor([[[[1 + 1j, 0.5 + 2j], [0.25 + 0.5j, 1.5 + 0.25j]]]], dtype=torch.complex64)
        gt = torch.tensor([[[[1 + 0j, 2 + 0.5j], [0.5 + 0.5j, 0.75 + 1.0j]]]], dtype=torch.complex64)
        target = {"image": torch.zeros(1, 1, 2, 2), "kspace": gt}

        criterion = PerpendicularLoss(magnitude_weighting=True, magnitude_weight_m=2.5, magnitude_weight_k=0.103, magnitude_weight_p=67.0)
        actual = criterion(pred, target)

        cross = pred * gt.conj()
        phi_hat = torch.angle(cross)
        perp = 0.5 * torch.abs(pred * gt.conj() - pred.conj() * gt) / (pred.abs() + criterion.eps)
        target_abs = gt.abs()
        branched = torch.where(phi_hat.abs() < (torch.pi / 2), perp, 2 * target_abs - perp)
        magnitude_l1 = torch.abs(target_abs - pred.abs())
        expected = torch.mean(branched + _perpendicular_mag_weight_map(pred, 2.5, 0.103, 67.0) * magnitude_l1)
        self.assertAlmostEqual(actual.item(), expected.item(), places=6)

    def test_perpendicular_loss_works_in_normalized_kspace_modes_with_weighting(self):
        gt = torch.tensor([[[[1 + 2j, 0.5 + 0.25j]]]], dtype=torch.complex64)
        pred = torch.tensor([[[[1.2 + 1.8j, 0.25 + 0.4j]]]], dtype=torch.complex64)
        target_comp = {"image": torch.zeros(1, 1, 1, 2), "kspace": apply_kspace_companding(gt, a=0.5, p=0.8)}
        target_log = {"image": torch.zeros(1, 1, 1, 2), "kspace": apply_log_kspace(gt)}
        pred_comp = apply_kspace_companding(pred, a=0.5, p=0.8)
        pred_log = apply_log_kspace(pred)
        criterion = PerpendicularLoss(magnitude_weighting=True, magnitude_weight_m=2.0)

        comp_loss = criterion(pred_comp, target_comp, stats={"normalization": "kspace_companding"})
        log_loss = criterion(pred_log, target_log, stats={"normalization": "log_kspace"})
        self.assertTrue(torch.isfinite(comp_loss))
        self.assertTrue(torch.isfinite(log_loss))

    def test_lambda_scheduler_can_schedule_perpendicular_m(self):
        constant = LambdaScheduler("constant", 2.0, 5.0, 5)
        linear = LambdaScheduler("linear", 2.0, 5.0, 5)
        cosine = LambdaScheduler("cosine", 2.0, 5.0, 5)

        self.assertAlmostEqual(constant.get_lambda(0), 2.0, places=6)
        self.assertAlmostEqual(constant.get_lambda(4), 2.0, places=6)
        self.assertAlmostEqual(linear.get_lambda(0), 2.0, places=6)
        self.assertAlmostEqual(linear.get_lambda(4), 5.0, places=6)
        self.assertAlmostEqual(cosine.get_lambda(0), 2.0, places=6)
        self.assertAlmostEqual(cosine.get_lambda(4), 5.0, places=6)

    def test_hard_data_consistency_does_not_create_lambda_scheduler(self):
        hard_cfg = types.SimpleNamespace(lambda_schedule="hard", lambda_start=0.0, lambda_end=0.0, epochs=400)
        constant_cfg = types.SimpleNamespace(lambda_schedule="constant", lambda_start=0.0, lambda_end=0.0, epochs=400)

        self.assertIsNone(_build_lambda_scheduler(hard_cfg))
        self.assertIsInstance(_build_lambda_scheduler(constant_cfg), LambdaScheduler)

    def test_kspace_losses_reject_real_predictions(self):
        pred = torch.ones(1, 1, 2, 2)
        target = {"image": torch.ones(1, 1, 2, 2), "kspace": torch.ones(1, 1, 2, 2, dtype=torch.complex64)}

        with self.assertRaisesRegex(ValueError, "complex prediction"):
            build_loss("complex_l1")(pred, target)
        with self.assertRaisesRegex(ValueError, "complex prediction"):
            build_loss("complex_l2")(pred, target)
        with self.assertRaisesRegex(ValueError, "complex prediction"):
            build_loss("perpendicular_loss")(pred, target)

    def test_cascade_forward_default_contract_is_unchanged(self):
        model = cascadeNet(
            N=(2, 2),
            encList=[_DummyEncoder, _DummyEncoder],
            encArgs=[{"delta": 0.0}, {"delta": 0.0}],
            lamb=False,
            learning="k_space",
        )
        x = torch.ones(1, 1, 2, 2, dtype=torch.cfloat)
        y = torch.ones(1, 1, 2, 2, dtype=torch.cfloat)
        mask = torch.ones(1, 1, 2, 2)

        recon = model(x, y, mask)

        self.assertTrue(torch.is_tensor(recon))
        self.assertTrue(recon.is_complex())

    def test_cascade_returns_complex_post_dc_intermediates_for_kspace(self):
        model = cascadeNet(
            N=(2, 2),
            encList=[_DummyEncoder, _DummyEncoder, _DummyEncoder],
            encArgs=[{"delta": 0.0}, {"delta": 0.0}, {"delta": 0.0}],
            lamb=False,
            learning="k_space",
        )
        x = torch.randn(1, 1, 2, 2, dtype=torch.cfloat)
        y = torch.randn(1, 1, 2, 2, dtype=torch.cfloat)
        mask = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])

        recon, intermediates = model(x, y, mask, return_intermediates=True)

        self.assertEqual(len(intermediates), 3)
        self.assertTrue(recon.is_complex())
        self.assertTrue(all(stage.is_complex() for stage in intermediates))

    def test_cascade_returns_real_post_dc_intermediates_for_image_mode(self):
        model = cascadeNet(
            N=(2, 2),
            encList=[_DummyEncoder, _DummyEncoder],
            encArgs=[{"delta": 0.0}, {"delta": 0.0}],
            lamb=False,
            learning="image",
        )
        x = torch.ones(1, 1, 2, 2)
        y = torch.randn(1, 1, 2, 2, dtype=torch.cfloat)
        mask = torch.ones(1, 1, 2, 2)

        recon, intermediates = model(x, y, mask, return_intermediates=True)

        self.assertFalse(recon.is_complex())
        self.assertEqual(len(intermediates), 2)
        self.assertTrue(all(not stage.is_complex() for stage in intermediates))

    def test_cascade_preserves_complex_image_outputs_through_fft_data_consistency(self):
        model = cascadeNet(
            N=(2, 2),
            encList=[_DummyEncoder, _DummyEncoder],
            encArgs=[{"delta": 0.0}, {"delta": 0.0}],
            lamb=False,
            learning="complex_image",
        )
        x = torch.randn(1, 1, 2, 2, dtype=torch.cfloat)
        y = torch.randn(1, 1, 2, 2, dtype=torch.cfloat)
        mask = torch.ones(1, 1, 2, 2)

        recon, intermediates = model(x, y, mask, return_intermediates=True)

        expected = centered_ifft2(y)
        self.assertTrue(recon.is_complex())
        self.assertTrue(torch.allclose(recon, expected, atol=1e-6, rtol=1e-6))
        self.assertTrue(all(stage.is_complex() for stage in intermediates))

    def test_train_config_contains_reconformer_experiment(self):
        reconformer_indices = [
            index for index, experiment in enumerate(EXPERIMENTS)
            if experiment.get("model_type") == "reconformer"
        ]
        self.assertEqual(len(reconformer_indices), 1)
        reconformer_cfg = build_cfg(reconformer_indices[0])
        self.assertEqual(reconformer_cfg.dataset, "fastmri")
        self.assertEqual(reconformer_cfg.model_type, "reconformer")
        self.assertEqual(reconformer_cfg.learning, "complex_image")
        self.assertEqual(reconformer_cfg.norm, "reconformer")
        self.assertEqual(reconformer_cfg.final_loss_type, "reconformer_l1")
        self.assertEqual(reconformer_cfg.scheduler_type, "step")
        self.assertEqual(reconformer_cfg.checkpoint_metric, "volume_psnr")
        self.assertIn(reconformer_cfg.hpc_backend, {"nvidia", "amd"})

    def test_p95_experiments_run_forward_loss_and_backward(self):
        experiment_indices = [
            index for index, experiment in enumerate(EXPERIMENTS)
            if experiment.get("model_type", "dctnn") == "dctnn"
            and experiment.get("norm", "zscore") == "fastmri_magnitude"
        ]
        for experiment_index in experiment_indices:
            with self.subTest(experiment_index=experiment_index):
                cfg = build_cfg(experiment_index)
                cfg.image_size = (4, 4)
                cfg.encoders = ["patch"]
                cfg.patch_size = (2, 2)
                cfg.nhead_patch = 1
                cfg.layer_no = 1
                cfg.num_encoder_layers = 1
                model = build_model(cfg)
                image = torch.randn(1, 1, 4, 4, dtype=torch.complex64)
                kspace = centered_fft2(image)
                mask = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])
                kspace_us = kspace * mask
                model_input, dc_input, target, stats = simulate_undersampling(
                    kspace,
                    mask,
                    learning=cfg.learning,
                    norm=cfg.norm,
                    kspace_us=kspace_us,
                )

                recon = model(model_input, dc_input, mask, stats=stats)
                loss = build_loss(cfg.final_loss_type)(recon, target, stats=stats)
                loss.backward()
                grads = [param.grad for param in model.parameters() if param.requires_grad]

                self.assertEqual(stats["normalization_domain"], cfg.learning)
                if cfg.learning == "k_space":
                    self.assertTrue(torch.allclose(model_input * mask, target["kspace"] * mask))
                    self.assertEqual(torch.count_nonzero(model_input * (1 - mask)).item(), 0)
                else:
                    expected_target = centered_ifft2(kspace) / stats["p95"]
                    self.assertTrue(torch.allclose(target["complex_image"], expected_target))
                raw_recon = model_output_to_raw_kspace(recon, stats, cfg.learning)
                self.assertTrue(torch.allclose(raw_recon * mask, dc_input * mask, atol=1e-6, rtol=1e-6))
                self.assertTrue(recon.is_complex())
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(grad is None or torch.isfinite(grad).all() for grad in grads))
                self.assertTrue(any(grad is not None and torch.any(grad != 0) for grad in grads))

    def test_oasis_dataset_behavior_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            arr = np.array([[0, 64], [128, 255]], dtype=np.uint8)
            from PIL import Image
            Image.fromarray(arr).save(Path(tmpdir) / "sample.png")

            ds = OASISDataset(tmpdir, image_size=(2, 2))
            sample = ds[0]
            img_t = torch.tensor(arr.astype('float32') / 255.0)
            expected = torch.fft.fftshift(
                torch.fft.fft2(torch.fft.ifftshift(img_t), norm='ortho')
            ).unsqueeze(0).to(torch.complex64)
            self.assertTrue(torch.allclose(sample, expected, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
