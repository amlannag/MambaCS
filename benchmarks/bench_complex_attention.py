import time

import torch

from DcTNN.attention_layer import ComplexMultiHeadAttention, TritonComplexMultiHeadAttention


def benchmark(module, x, warmup=10, steps=50):
    for _ in range(warmup):
        y = module(x)
        loss = y.real.mean() + y.imag.mean()
        loss.backward()
        module.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(steps):
        y = module(x)
        loss = y.real.mean() + y.imag.mean()
        loss.backward()
        module.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / steps


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the complex attention benchmark.")

    torch.manual_seed(0)
    b, n, c, h = 2, 256, 128, 8
    x = torch.randn(b, n, c, dtype=torch.cfloat, device="cuda", requires_grad=True)

    ref = ComplexMultiHeadAttention(d_model=c, nhead=h, dropout=0.0).cuda().train()
    tri = TritonComplexMultiHeadAttention(d_model=c, nhead=h, dropout=0.0).cuda().train()
    tri.qkv.load_state_dict(ref.qkv.state_dict())
    tri.proj.load_state_dict(ref.proj.state_dict())

    ref_t = benchmark(ref, x.clone())
    tri_t = benchmark(tri, x.clone())

    print(f"reference_complex: {ref_t * 1e3:.3f} ms/iter")
    print(f"triton_complex   : {tri_t * 1e3:.3f} ms/iter")
    print(f"speedup          : {ref_t / tri_t:.3f}x")


if __name__ == "__main__":
    main()
