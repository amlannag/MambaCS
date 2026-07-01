import unittest

import torch

from DcTNN.attention_layer import ComplexMultiHeadAttention, TritonComplexMultiHeadAttention


class ComplexTritonAttentionTest(unittest.TestCase):
    def test_requires_cuda(self):
        mod = TritonComplexMultiHeadAttention(d_model=32, nhead=4, dropout=0.0)
        x = torch.randn(2, 8, 32, dtype=torch.cfloat)
        with self.assertRaises(RuntimeError):
            mod(x)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton attention tests.")
    def test_matches_reference_eval(self):
        torch.manual_seed(0)
        ref = ComplexMultiHeadAttention(d_model=32, nhead=4, dropout=0.0).cuda().eval()
        tri = TritonComplexMultiHeadAttention(d_model=32, nhead=4, dropout=0.0).cuda().eval()
        tri.qkv.load_state_dict(ref.qkv.state_dict())
        tri.proj.load_state_dict(ref.proj.state_dict())

        x = torch.randn(2, 16, 32, dtype=torch.cfloat, device="cuda")
        y_ref = ref(x)
        y_tri = tri(x)

        self.assertEqual(y_ref.shape, y_tri.shape)
        self.assertTrue(torch.allclose(y_ref, y_tri, atol=1e-5, rtol=1e-4))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton attention tests.")
    def test_matches_reference_train(self):
        torch.manual_seed(1234)
        ref = ComplexMultiHeadAttention(d_model=32, nhead=4, dropout=0.1).cuda().train()
        tri = TritonComplexMultiHeadAttention(d_model=32, nhead=4, dropout=0.1).cuda().train()
        tri.qkv.load_state_dict(ref.qkv.state_dict())
        tri.proj.load_state_dict(ref.proj.state_dict())

        x = torch.randn(2, 16, 32, dtype=torch.cfloat, device="cuda")
        torch.manual_seed(2024)
        y_ref = ref(x)
        torch.manual_seed(2024)
        y_tri = tri(x)

        self.assertEqual(y_ref.shape, y_tri.shape)
        self.assertTrue(torch.allclose(y_ref, y_tri, atol=1e-5, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()
