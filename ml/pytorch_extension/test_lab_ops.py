#!/usr/bin/env python3
"""Build lab_ops.cu as a PyTorch extension, check it against PyTorch, and time both.

Requires PyTorch with CUDA and a GPU:
    pip install torch
    python3 ml/pytorch_extension/test_lab_ops.py

torch.utils.cpp_extension.load compiles the extension on first use (this takes
a minute) and caches it. For a packaged build, the same file works with
torch.utils.cpp_extension.CUDAExtension in a setup.py.
"""
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def rms_norm_reference(x, weight, eps):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight


def time_ms(fn, reps=50):
    for _ in range(5):
        fn()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / reps


def main():
    if not torch.cuda.is_available():
        raise SystemExit("SKIP: no CUDA device")
    here = Path(__file__).resolve().parent
    lab_ops = load(name="lab_ops", sources=[str(here / "lab_ops.cu")], verbose=False)

    torch.manual_seed(0)
    x = torch.randn(8192, 4096, device="cuda")
    weight = torch.rand(4096, device="cuda") + 0.5
    eps = 1e-5

    ok = True
    for name, ours, theirs in [
        ("rms_norm", lambda: lab_ops.rms_norm(x, weight, eps), lambda: rms_norm_reference(x, weight, eps)),
        ("softmax", lambda: lab_ops.softmax(x), lambda: torch.softmax(x, dim=-1)),
    ]:
        match = torch.allclose(ours(), theirs(), rtol=1e-4, atol=1e-6)
        ok &= match
        print(f"{name:9s} matches PyTorch: {match}   "
              f"lab_ops {time_ms(ours):.3f} ms   PyTorch {time_ms(theirs):.3f} ms")
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
