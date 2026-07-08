import sys
import types
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

wandb_stub = types.SimpleNamespace(init=lambda *a, **k: None, log=lambda *a, **k: None, finish=lambda *a, **k: None)
sys.modules.setdefault("wandb", wandb_stub)

from DcTNN.loss import MagnitudeImageLoss, MagnitudeL1Loss, build_loss
from DcTNN.model import cascadeNet
from train import _compute_losses


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
        with self.assertRaisesRegex(ValueError, "Unknown loss_type"):
            build_loss("bad")

    def test_compute_losses_final_only_logs_intermediates_without_optimizing_them(self):
        gt = torch.ones(1, 1, 2, 2)
        recon = torch.full((1, 1, 2, 2), 2.0)
        intermediates = [torch.full((1, 1, 2, 2), 3.0), torch.full((1, 1, 2, 2), 4.0)]

        total_loss, final_loss, intermediate_loss_sum, stage_losses = _compute_losses(
            recon, intermediates, gt, build_loss("l1"), build_loss("l2"), "final_only"
        )

        self.assertAlmostEqual(final_loss.item(), 1.0, places=6)
        self.assertAlmostEqual(stage_losses[0].item(), 4.0, places=6)
        self.assertAlmostEqual(stage_losses[1].item(), 9.0, places=6)
        self.assertAlmostEqual(intermediate_loss_sum.item(), 13.0, places=6)
        self.assertAlmostEqual(total_loss.item(), final_loss.item(), places=6)

    def test_compute_losses_intermediate_unweighted_sums_stage_losses(self):
        gt = torch.zeros(1, 1, 2, 2)
        recon = torch.ones(1, 1, 2, 2)
        intermediates = [torch.ones(1, 1, 2, 2), torch.full((1, 1, 2, 2), 2.0)]

        total_loss, final_loss, intermediate_loss_sum, stage_losses = _compute_losses(
            recon, intermediates, gt, build_loss("l2"), build_loss("l1"), "intermediate_unweighted"
        )

        self.assertAlmostEqual(final_loss.item(), 1.0, places=6)
        self.assertAlmostEqual(stage_losses[0].item(), 1.0, places=6)
        self.assertAlmostEqual(stage_losses[1].item(), 2.0, places=6)
        self.assertAlmostEqual(intermediate_loss_sum.item(), 3.0, places=6)
        self.assertAlmostEqual(total_loss.item(), 4.0, places=6)

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
