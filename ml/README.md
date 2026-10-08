# Machine-Learning Kernels

The kernels at the heart of a transformer, written from scratch. Steps 00-03
are row-wise and bandwidth-bound (they share
[rowwise_harness.cuh](rowwise_harness.cuh); % of peak bandwidth is the figure
of merit); steps 04-05 compute attention (they share
[attention_harness.cuh](attention_harness.cuh)).

| Step | What it computes | The idea |
| --- | --- | --- |
| [00_softmax_naive](00_softmax_naive/00_softmax_naive.cu) | Row softmax | Three passes: max, sum of exponentials, normalize |
| [01_softmax_online](01_softmax_online/01_softmax_online.cu) | Row softmax | Online max and sum in one pass, rescaling the sum when the max grows (the trick FlashAttention is built on) |
| [02_layernorm](02_layernorm/02_layernorm.cu) | LayerNorm | Welford mean and variance, merged exactly across threads |
| [03_rmsnorm](03_rmsnorm/03_rmsnorm.cu) | RMSNorm (LLaMA-style) | One statistic, a plain sum of squares |
| [04_attention_naive](04_attention_naive/04_attention_naive.cu) | softmax(Q K^T / sqrt(d)) V | Materializes the n x n score matrix in global memory |
| [05_flash_attention](05_flash_attention/05_flash_attention.cu) | The same output | FlashAttention forward: Q tile on chip, K/V streamed in tiles, online softmax; the n x n matrix never exists |

All kernels use float4 accesses where they can and validate against a
double-precision CPU reference.

## PyTorch extension

[pytorch_extension/](pytorch_extension/) wraps the RMSNorm and online softmax
kernels as a PyTorch C++/CUDA extension and compares them with PyTorch's own
operators:

```bash
pip install torch
python3 ml/pytorch_extension/test_lab_ops.py
```

The test compiles the extension on first use with
`torch.utils.cpp_extension.load`. It needs PyTorch and a GPU, so CI only
checks that the script parses.
