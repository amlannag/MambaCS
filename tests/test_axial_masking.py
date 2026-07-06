import sys
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from DcTNN.attention_layer import ComplexCrossAttention
from DcTNN.encoders import (
    axialEncoder,
    crossAxialEncoder,
    _CrossAxialInnerLayer,
    _build_vertical_attn_mask,
    _extract_shared_column_mask,
)
from DcTNN.rope_vit import compute_axial_cis_complex
from DcTNN.vit import CrossAttentionVIT


class RecorderSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_mask = None
        self.last_positions = None
        self.calls = 0

    def forward(self, x, attn_mask=None, positions=None):
        self.calls += 1
        self.last_mask = None if attn_mask is None else attn_mask.clone()
        self.last_positions = None if positions is None else positions.clone()
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


class RecorderCrossAxialStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.last_x = None
        self.last_sampled_idx = None
        self.last_unsampled_idx = None

    def forward(self, x, sampled_idx, unsampled_idx):
        self.calls += 1
        self.last_x = x.clone()
        self.last_sampled_idx = sampled_idx.clone()
        self.last_unsampled_idx = unsampled_idx.clone()
        out = x.clone()
        if unsampled_idx.numel() > 0:
            out[:, unsampled_idx, :] = out[:, unsampled_idx, :] + 7.0
        return out


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

    def test_axial_encoder_rejects_cross_mode(self):
        with self.assertRaisesRegex(ValueError, "Use the 'cross_axial' encoder family instead"):
            axialEncoder(
                image_size=(4, 4),
                numCh=1,
                d_model=2,
                nhead=1,
                num_layers=2,
                dim_feedforward=4,
                dropout=0.0,
                pos_emb_type="Rope-Axial",
                attn_type="complex",
                mask_vertical_attn="cross",
            )

    def _make_stub_axial_encoder(self, mode):
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
        enc = self._make_stub_axial_encoder("none")
        recorder = RecorderSelfAttention()
        enc.verticalEncoder = recorder

        x = torch.randn(1, 4, 2, dtype=torch.cfloat)
        mask = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])
        out = enc(x, col_mask=mask)

        self.assertEqual(recorder.calls, 1)
        self.assertIsNone(recorder.last_mask)
        self.assertTrue(torch.equal(out, x))

    def test_lenient_passes_expected_mask(self):
        enc = self._make_stub_axial_encoder("lenient")
        recorder = RecorderSelfAttention()
        enc.verticalEncoder = recorder

        x = torch.randn(1, 4, 2, dtype=torch.cfloat)
        mask = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])
        enc(x, col_mask=mask)

        expected = _build_vertical_attn_mask(torch.tensor([True, False, True, False]), "lenient")
        self.assertTrue(torch.equal(recorder.last_mask, expected))

    def test_strict_passes_expected_mask(self):
        enc = self._make_stub_axial_encoder("strict")
        recorder = RecorderSelfAttention()
        enc.verticalEncoder = recorder

        x = torch.randn(1, 4, 2, dtype=torch.cfloat)
        mask = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])
        enc(x, col_mask=mask)

        expected = _build_vertical_attn_mask(torch.tensor([True, False, True, False]), "strict")
        self.assertTrue(torch.equal(recorder.last_mask, expected))


class CrossAxialEncoderTest(unittest.TestCase):
    def test_cross_axial_rejects_non_complex_attention(self):
        with self.assertRaisesRegex(ValueError, "only supports attn_type='complex'"):
            crossAxialEncoder(
                image_size=(4, 4),
                numCh=1,
                d_model=2,
                nhead=1,
                num_layers=2,
                dim_feedforward=4,
                dropout=0.0,
                pos_emb_type="Rope-Axial",
                attn_type="standard",
            )

    def test_cross_inner_layer_routes_cross_then_unsampled_self_twice(self):
        layer = _CrossAxialInnerLayer(
            d_model=2, nhead=1, dim_feedforward=4, dropout=0.0,
            activation="relu", layer_norm_eps=1e-5, attn_type="complex"
        )
        cross1 = RecorderCrossAttention(delta=3.0)
        self_attn1 = RecorderSelfAttention()
        cross2 = RecorderCrossAttention(delta=5.0)
        self_attn2 = RecorderSelfAttention()
        layer.cross1 = cross1
        layer.self_attn1 = self_attn1
        layer.cross2 = cross2
        layer.self_attn2 = self_attn2

        x = torch.tensor(
            [[[0.0 + 0.0j, 1.0 + 0.0j],
              [2.0 + 0.0j, 3.0 + 0.0j],
              [4.0 + 0.0j, 5.0 + 0.0j],
              [6.0 + 0.0j, 7.0 + 0.0j]]],
            dtype=torch.cfloat,
        )
        sampled_idx = torch.tensor([0, 2])
        unsampled_idx = torch.tensor([1, 3])
        out = layer(x, sampled_idx, unsampled_idx)

        self.assertEqual(cross1.calls, 1)
        self.assertEqual(cross2.calls, 1)
        self.assertTrue(torch.equal(cross1.last_q_positions, unsampled_idx))
        self.assertTrue(torch.equal(cross1.last_kv_positions, sampled_idx))
        self.assertTrue(torch.equal(cross1.last_q, x[:, [1, 3], :]))
        self.assertTrue(torch.equal(cross1.last_kv, x[:, [0, 2], :]))
        self.assertTrue(torch.equal(cross2.last_q_positions, unsampled_idx))
        self.assertTrue(torch.equal(cross2.last_kv_positions, sampled_idx))
        self.assertEqual(self_attn1.calls, 1)
        self.assertEqual(self_attn2.calls, 1)
        self.assertTrue(torch.equal(self_attn1.last_positions, unsampled_idx))
        self.assertTrue(torch.equal(self_attn2.last_positions, unsampled_idx))
        self.assertTrue(torch.equal(out[:, [0, 2], :], x[:, [0, 2], :]))
        self.assertTrue(torch.equal(out[:, [1, 3], :], x[:, [1, 3], :] + 8.0))

    def test_cross_axial_encoder_updates_unsampled_and_preserves_sampled_before_head(self):
        enc = crossAxialEncoder(
            image_size=(4, 4),
            numCh=1,
            d_model=2,
            nhead=1,
            num_layers=2,
            dim_feedforward=4,
            dropout=0.0,
            pos_emb_type="Rope-Axial",
            attn_type="complex",
        )
        enc.to_vertical_embedding = nn.Identity()
        enc.dropout = nn.Identity()
        enc.vertical_mlp_head = nn.Identity()
        recorder = RecorderCrossAxialStack()
        enc.verticalEncoder = recorder

        x = torch.tensor(
            [[[0.0 + 0.0j, 1.0 + 0.0j],
              [2.0 + 0.0j, 3.0 + 0.0j],
              [4.0 + 0.0j, 5.0 + 0.0j],
              [6.0 + 0.0j, 7.0 + 0.0j]]],
            dtype=torch.cfloat,
        )
        mask = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])
        out = enc(x, col_mask=mask)

        self.assertEqual(recorder.calls, 1)
        self.assertTrue(torch.equal(recorder.last_sampled_idx, torch.tensor([0, 2])))
        self.assertTrue(torch.equal(recorder.last_unsampled_idx, torch.tensor([1, 3])))
        self.assertTrue(torch.equal(out[:, [0, 2], :], x[:, [0, 2], :]))
        self.assertTrue(torch.equal(out[:, [1, 3], :], x[:, [1, 3], :] + 7.0))

    def test_cross_axial_requires_mask(self):
        enc = crossAxialEncoder(
            image_size=(4, 4),
            numCh=1,
            d_model=2,
            nhead=1,
            num_layers=2,
            dim_feedforward=4,
            dropout=0.0,
            pos_emb_type="Rope-Axial",
            attn_type="complex",
        )
        x = torch.randn(1, 1, 4, 4, dtype=torch.cfloat)
        with self.assertRaisesRegex(ValueError, "requires col_mask"):
            enc(x, col_mask=None)

    def test_cross_attention_vit_constructs(self):
        vit = CrossAttentionVIT(
            N=(4, 4),
            layerNo=2,
            numCh=1,
            d_model=4,
            nhead=1,
            num_encoder_layers=2,
            dim_feedforward=8,
            dropout=0.0,
            pos_emb_type="Rope-Axial",
            attn_type="complex",
        )
        self.assertEqual(len(vit.transformers), 2)
        self.assertTrue(all(isinstance(layer, crossAxialEncoder) for layer in vit.transformers))


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
