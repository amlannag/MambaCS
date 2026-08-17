import sys
import types
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

wandb_stub = types.SimpleNamespace(init=lambda *a, **k: None, log=lambda *a, **k: None, finish=lambda *a, **k: None)
sys.modules.setdefault("wandb", wandb_stub)

from DcTNN.loss import (
    ComplexL1Loss,
    ComplexL2Loss,
    MagnitudeImageLoss,
    MagnitudeL1Loss,
    PerpendicularLoss,
    build_loss,
)
from DcTNN.model import cascadeNet
from inference import _denormalize_image, to_image_magnitude
from normalizer import (
    apply_kspace_companding,
    apply_log_kspace,
    invert_kspace_companding,
    invert_log_kspace,
    kspace_to_image_magnitude,
    restore_original_kspace,
)
from train import _compute_losses
from train_utils import simulate_undersampling


class _DummyEncoder(nn.Module):
    def __init__(self, image_size, delta=0.0, **kwargs):
        super().__init__()
        self.delta = delta

    def forward(self, x, col_mask=None):
        return torch.full_like(x, self.delta)


class IntermediateLossTest(unittest.TestCase):
    def test_build_loss_supports_l1_and_l2(self):
        self.assertIsInstance(build_loss("l1"), MagnitudeL1Loss)
        self.assertIsInstance(build_loss("l2"), MagnitudeImageLoss)
        self.assertIsInstance(build_loss("image_domain_l1"), MagnitudeL1Loss)
        self.assertIsInstance(build_loss("image_domain_l2"), MagnitudeImageLoss)
        self.assertIsInstance(build_loss("complex_l1"), ComplexL1Loss)
        self.assertIsInstance(build_loss("complex_l2"), ComplexL2Loss)
        self.assertIsInstance(build_loss("perpendicular_loss"), PerpendicularLoss)
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

    def test_kspace_companding_forward_inverse_recovers_input_and_preserves_phase(self):
        real = torch.tensor([[[[1.0, -2.0], [0.5, -0.25]]]])
        imag = torch.tensor([[[[0.25, 1.5], [-0.75, 2.0]]]])
        ks = torch.complex(real, imag)

        companded = apply_kspace_companding(ks, a=0.5, p=0.8)
        restored = invert_kspace_companding(companded, a=0.5, p=0.8)

        self.assertTrue(torch.allclose(restored, ks, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(torch.angle(companded), torch.angle(ks), atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.isfinite(companded.abs()).all())

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
        self.assertTrue(torch.allclose(model_input, apply_kspace_companding(kspace_full, a=0.5, p=0.8), atol=1e-6, rtol=1e-6))
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
        stats = {"normalization": "kspace_companding", "companding_p": 0.8, "companding_a": 0.5}

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
        comp_stats = {"normalization": "kspace_companding", "companding_p": 0.8, "companding_a": 0.5}
        log_stats = {"normalization": "log_kspace"}

        self.assertTrue(torch.allclose(restore_original_kspace(apply_kspace_companding(ks, 0.5, 0.8), comp_stats), ks, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(restore_original_kspace(apply_log_kspace(ks), log_stats), ks, atol=1e-5, rtol=1e-5))

    def test_inference_denormalize_returns_original_kspace_for_supported_kspace_norms(self):
        ks = torch.tensor([[[[1 + 2j, 0.5 + 0.25j]]]], dtype=torch.complex64)
        comp_stats = {"normalization": "kspace_companding", "companding_p": 0.8, "companding_a": 0.5}
        log_stats = {"normalization": "log_kspace"}

        comp = apply_kspace_companding(ks, a=0.5, p=0.8)
        logged = apply_log_kspace(ks)

        self.assertTrue(torch.allclose(_denormalize_image(comp, comp_stats), ks, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(_denormalize_image(logged, log_stats), ks, atol=1e-5, rtol=1e-5))

    def test_to_image_magnitude_restores_kspace_before_ifft(self):
        img = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)
        kspace = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(img), norm='ortho')).to(torch.complex64)
        comp_stats = {"normalization": "kspace_companding", "companding_p": 0.8, "companding_a": 0.5}
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

    def test_kspace_losses_reject_real_predictions(self):
        pred = torch.ones(1, 1, 2, 2)
        target = {"image": torch.ones(1, 1, 2, 2), "kspace": torch.ones(1, 1, 2, 2, dtype=torch.complex64)}

        with self.assertRaisesRegex(ValueError, "complex k-space"):
            build_loss("complex_l1")(pred, target)
        with self.assertRaisesRegex(ValueError, "complex k-space"):
            build_loss("complex_l2")(pred, target)
        with self.assertRaisesRegex(ValueError, "complex k-space"):
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


if __name__ == "__main__":
    unittest.main()
