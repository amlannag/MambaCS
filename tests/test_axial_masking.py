import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, "/Users/amlannag/Desktop/MambaCS")

from DcTNN.attention_layer import ComplexCrossAttention
from DcTNN.encoders import axialEncoder, _build_vertical_attn_mask, _extract_shared_column_mask
from DcTNN.rope_vit import compute_axial_cis_complex


class RecorderSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_mask = None
        self.calls = 0

    def forward(self, x, attn_mask=None):
        self.calls += 1
        self.last_mask = None if attn_mask is None else attn_mask.clone()
        return x


class RecorderCrossAttention(nn.Module):
    def __init__(self, delta=10.0):
        super().__init__()
        self.delta = delta
        self.calls = 0
        self.last_q = None
        self.last_kv = None
        self.last_q_positions = None
        self.last_kv_positions = None

    def forward(self, q, kv, attn_mask=None, q_positions=None, kv_positions=None):
        self.calls += 1
        self.last_q = q.clone()
        self.last_kv = kv.clone()
        self.last_q_positions = None if q_positions is None else q_positions.clone()
        self.last_kv_positions = None if kv_positions is None else kv_positions.clone()
        return q + torch.full_like(q, self.delta)


class UnexpectedCall(nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("This module should not have been called")


class AxialMaskingTest(unittest.TestCase):
    def test_build_vertical_attn_mask_none(self):
        sampled = torch.tensor([True, False, True, False])
        mask = _build_vertical_attn_mask(sampled, "none")
        self.assertEqual(mask.shape, (4, 4))
        self.assertTrue(torch.equal(mask, torch.zeros(4, 4)))

    def test_build_vertical_attn_mask_lenient(self):
        sampled = torch.tensor([True, False, True, False])
        mask = _build_vertical_attn_mask(sampled, "lenient")
        expected = torch.tensor([
            [0.0, float("-inf"), 0.0, float("-inf")],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, float("-inf"), 0.0, float("-inf")],
            [0.0, 0.0, 0.0, 0.0],
        ])
        self.assertTrue(torch.equal(mask, expected))

    def test_build_vertical_attn_mask_strict(self):
        sampled = torch.tensor([True, False, True, False])
        mask = _build_vertical_attn_mask(sampled, "strict")
        expected = torch.tensor([
            [0.0, float("-inf"), 0.0, float("-inf")],
            [0.0, float("-inf"), 0.0, float("-inf")],
            [0.0, float("-inf"), 0.0, float("-inf")],
            [0.0, float("-inf"), 0.0, float("-inf")],
        ])
        self.assertTrue(torch.equal(mask, expected))

    def test_extract_shared_column_mask_accepts_broadcast_mask(self):
        mask = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])
        sampled = _extract_shared_column_mask(mask, width=4)
        self.assertTrue(torch.equal(sampled, torch.tensor([True, False, True, False])))

    def test_extract_shared_column_mask_rejects_per_example_masks(self):
        mask = torch.tensor([
            [[[1.0, 0.0, 1.0, 0.0]]],
            [[[1.0, 1.0, 0.0, 0.0]]],
        ])
        with self.assertRaisesRegex(ValueError, "Per-example column masks are not supported"):
            _extract_shared_column_mask(mask, width=4)

    def _make_stub_encoder(self, mode):
        enc = axialEncoder(
            image_size=(4, 4),
            numCh=1,
            d_model=2,
            nhead=1,
            num_layers=2,
            dim_feedforward=4,
            dropout=0.0,
            pos_emb_type="Rope-Axial",
            attn_type="complex",
            mask_vertical_attn=mode,
        )
        enc.to_horizontal_embedding = nn.Identity()
        enc.horizontalEncoder = nn.Identity()
        enc.horizontal_mlp_head = nn.Identity()
        enc.to_vertical_embedding = nn.Identity()
        enc.dropout = nn.Identity()
        enc.vertical_mlp_head = nn.Identity()
        return enc

    def test_none_uses_unmasked_vertical_attention(self):
        enc = self._make_stub_encoder("none")
        recorder = RecorderSelfAttention()
        enc.verticalEncoder = recorder

        x = torch.randn(1, 4, 2, dtype=torch.cfloat)
        mask = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])
        out = enc(x, col_mask=mask)

        self.assertEqual(recorder.calls, 1)
        self.assertIsNone(recorder.last_mask)
        self.assertTrue(torch.equal(out, x))

    def test_lenient_passes_expected_mask(self):
        enc = self._make_stub_encoder("lenient")
        recorder = RecorderSelfAttention()
        enc.verticalEncoder = recorder

        x = torch.randn(1, 4, 2, dtype=torch.cfloat)
        mask = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])
        enc(x, col_mask=mask)

        expected = _build_vertical_attn_mask(torch.tensor([True, False, True, False]), "lenient")
        self.assertTrue(torch.equal(recorder.last_mask, expected))

    def test_strict_passes_expected_mask(self):
        enc = self._make_stub_encoder("strict")
        recorder = RecorderSelfAttention()
        enc.verticalEncoder = recorder

        x = torch.randn(1, 4, 2, dtype=torch.cfloat)
        mask = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])
        enc(x, col_mask=mask)

        expected = _build_vertical_attn_mask(torch.tensor([True, False, True, False]), "strict")
        self.assertTrue(torch.equal(recorder.last_mask, expected))

    def test_cross_updates_only_unsampled_columns(self):
        enc = self._make_stub_encoder("cross")
        enc.verticalEncoder = UnexpectedCall()
        cross = RecorderCrossAttention(delta=10.0)
        enc.verticalCrossEncoder = cross

        x = torch.tensor(
            [[[0.0 + 0.0j, 1.0 + 0.0j],
              [2.0 + 0.0j, 3.0 + 0.0j],
              [4.0 + 0.0j, 5.0 + 0.0j],
              [6.0 + 0.0j, 7.0 + 0.0j]]],
            dtype=torch.cfloat,
        )
        mask = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])
        out = enc(x, col_mask=mask)

        self.assertEqual(cross.calls, 1)
        self.assertTrue(torch.equal(cross.last_q_positions, torch.tensor([1, 3])))
        self.assertTrue(torch.equal(cross.last_kv_positions, torch.tensor([0, 2])))
        self.assertTrue(torch.equal(cross.last_q, x[:, [1, 3], :]))
        self.assertTrue(torch.equal(cross.last_kv, x[:, [0, 2], :]))
        self.assertTrue(torch.equal(out[:, [0, 2], :], x[:, [0, 2], :]))
        self.assertTrue(torch.equal(out[:, [1, 3], :], x[:, [1, 3], :] + 10.0))


class ComplexCrossAttentionTest(unittest.TestCase):
    def test_shape_and_dtype_without_rope(self):
        torch.manual_seed(0)
        attn = ComplexCrossAttention(d_model=8, nhead=2, dropout=0.0).eval()
        q = torch.randn(2, 3, 8, dtype=torch.cfloat)
        kv = torch.randn(2, 4, 8, dtype=torch.cfloat)
        out = attn(q, kv)
        self.assertEqual(out.shape, q.shape)
        self.assertEqual(out.dtype, torch.cfloat)

    def test_shape_and_dtype_with_rope_subset_positions(self):
        torch.manual_seed(1)
        freqs = compute_axial_cis_complex(dim=4, end_x=4, end_y=1)
        attn = ComplexCrossAttention(d_model=8, nhead=2, dropout=0.0, freqs_cis=freqs).eval()
        q = torch.randn(2, 2, 8, dtype=torch.cfloat)
        kv = torch.randn(2, 2, 8, dtype=torch.cfloat)
        out = attn(q, kv, q_positions=torch.tensor([1, 3]), kv_positions=torch.tensor([0, 2]))
        self.assertEqual(out.shape, q.shape)
        self.assertEqual(out.dtype, torch.cfloat)


if __name__ == "__main__":
    unittest.main()
