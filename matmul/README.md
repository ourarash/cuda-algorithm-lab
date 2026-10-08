# CUDA Matrix Multiplication (GEMM) Ladder

Twelve steps from a textbook kernel to Hopper Tensor Cores. Every step computes
`C = alpha * A @ B + beta * C` for row-major matrices and changes one idea
from the step before.

Every stage shares one host-side driver, [gemm_harness.cuh](gemm_harness.cuh),
so each `.cu` file contains only its kernel and launch configuration. The
harness:

- validates against a double-precision CPU reference, with `beta != 0` so the
  read-modify-write of `C` is tested too;
- reports GFLOP/s from warmed-up, repeated runs;
- runs the same problem through cuBLAS and prints the stage's speed as a
  percentage of it (`cublasSgemm` for FP32 stages, `cublasGemmEx` with FP16
  inputs and FP32 accumulation for the Tensor Core stages);
- skips (exit code 77) on GPUs a stage cannot run on.

Pass `--quick` for small sizes that are not multiples of the tile sizes, or
`--m`, `--n`, `--k` to choose your own.

Stages 00-07 follow Simon Boehm's article
[How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM),
an excellent companion read.

## The ladder

| Step | What changes | Nsight Compute evidence |
| --- | --- | --- |
| [00_uncoalesced](00_uncoalesced/00_matmul_uncoalesced.cu) | One thread per output; `threadIdx.x` walks rows, so a warp's loads are scattered | Global load sectors per request (`l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` / `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`) far above the ideal 4 |
| [01_coalesced](01_coalesced/01_matmul_coalesced.cu) | `threadIdx.x` walks columns, so a warp reads consecutive addresses | Sectors per request drops to about 4 |
| [02_shared_memory](02_shared_memory/02_matmul_shared_memory.cu) | 32 x 32 tiles of A and B staged in shared memory, each element reused 32 times | Global traffic (`dram__bytes_read.sum`, L2 traffic) drops sharply |
| [03_1d_register_tiling](03_1d_register_tiling/03_matmul_1d_register_tiling.cu) | Each thread computes 8 outputs, reusing each B value from a register | Shared load wavefronts per FMA (`l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum`) drop |
| [04_2d_register_tiling](04_2d_register_tiling/04_matmul_2d_register_tiling.cu) | Each thread computes an 8 x 8 patch: 64 FMAs per 16 shared loads | Same metric, lower again; bank conflicts (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`) are now visible |
| [05_vectorized](05_vectorized/05_matmul_vectorized.cu) | `float4` global and shared accesses; A stored transposed in shared memory | Load instruction counts (`smsp__sass_inst_executed_op_global_ld.sum`, `smsp__sass_inst_executed_op_shared_ld.sum`) drop about 4x |
| [06_warptiling](06_warptiling/06_matmul_warptiling.cu) | A warp owns a compact 64 x 64 tile; 128 outputs per thread | Shared loads per FMA drop again; more FMA per issued instruction |
| [07_double_buffering](07_double_buffering/07_matmul_double_buffering.cu) | `cp.async` loads the next K tile while the current one is computed | Long-scoreboard and barrier stalls (`smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio`) drop |
| [08_tensor_cores_wmma](08_tensor_cores_wmma/08_matmul_wmma.cu) | Tensor Cores through WMMA, 16 x 16 fragments straight from global memory | Tensor pipe active (`sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active`) is non-zero but low |
| [09_wmma_shared_memory](09_wmma_shared_memory/09_matmul_wmma_shared_memory.cu) | 128 x 128 block tile staged in shared memory, 8 fragments per warp | Global traffic drops; tensor pipe utilization rises |
| [10_mma_sync](10_mma_sync/10_matmul_mma_sync.cu) | Raw PTX `mma.sync` + `ldmatrix`, `cp.async` double buffering, XOR-swizzled shared memory | Shared bank conflicts near zero; tensor pipe utilization rises again |
| [11_hopper_wgmma_tma](11_hopper_wgmma_tma/11_matmul_hopper_wgmma.cu) | Hopper only: TMA bulk copies and warpgroup `wgmma` reading operands from shared memory | Tensor pipe utilization in the Compute Workload Analysis section; no `ldmatrix` or per-thread copy instructions |

Steps 00-07 use FP32 throughout. Steps 08-11 use FP16 inputs with FP32
accumulation, which is what Tensor Cores need, so compare them with the FP16
cuBLAS baseline each one prints rather than with the FP32 steps.

## Notes on individual steps

**02, why there is no padding.** Bank conflicts only happen between threads
of one warp within one instruction. At a fixed `k`, a warp reads `tileA[ty][k]`
(one address, a broadcast) and `tileB[k][tx]` (32 consecutive words, 32
different banks), so neither tile needs padding. Padding matters when a warp
reads down a column at once; see `matrix_transpose/`.

**04, what limits 2D tiling.** Every shared-memory load is a separate 32-bit
instruction, and the reads have 2-way (A) and 4-way (B) bank conflicts; the
file explains where they come from. Step 05 widens the loads.

**05 requirements.** `N` and `K` must be multiples of 4 so every `float4` is
16-byte aligned. The same holds for 06 and 07.

**07 and the transposing copy.** `cp.async` copies contiguous 4, 8, or 16
bytes, so writing A transposed into shared memory needs 4-byte copies, which
cause bank conflicts on the writes. Step 10 avoids the transpose entirely.

**10, swizzling instead of padding.** The 16-byte chunks of each shared-memory
row are permuted with an XOR of the row index, so the 8 rows of every
`ldmatrix` read land in 8 different groups of banks, without wasting memory or
breaking 16-byte alignment. Needs compute capability 8.0+; `N` and `K` must be
multiples of 8.

**11, Hopper.** Needs compute capability 9.0 exactly (code for the `sm_90a`
target runs only on Hopper) and reads B "K-major", so the harness passes it B
transposed. This is the simplest correct TMA + WGMMA kernel: one shared-memory
stage and no overlap of loads and math. Production Hopper GEMMs add a
multi-stage pipeline with producer and consumer warpgroups, thread block
clusters, and TMA multicast. It is compiled in CI but has not yet been run on
Hopper hardware; the harness validates it on every run.

## Results

Measure on your GPU and paste the output here:

```bash
make bench       # results tables for every step, with % of cuBLAS
make roofline    # docs/roofline.png; uses Nsight Compute (ncu) if installed
```

No measurements are committed yet: CI runners have no GPU, and numbers depend
heavily on the GPU, so each table should name the GPU it came from.

## Visualizations

- [00_uncoalesced/naive_visualization.html](00_uncoalesced/naive_visualization.html): memory access patterns of the naive kernel
- [01_coalesced/01_thread_to_memory_mapping_visualization.html](01_coalesced/01_thread_to_memory_mapping_visualization.html): thread-to-memory mapping, coalesced vs. strided
- [02_shared_memory/02_matmul_shared_memory_visualization.html](02_shared_memory/02_matmul_shared_memory_visualization.html): shared-memory tiling step by step
- [03_1d_register_tiling/03_matmul_register_tiling_visualization.html](03_1d_register_tiling/03_matmul_register_tiling_visualization.html) and the [single-thread view](03_1d_register_tiling/03_matmul_register_tiling_visualization_single_thread.html): 1D register tiling
- [03_1d_register_tiling/outer-product.html](03_1d_register_tiling/outer-product.html): inner-product vs. outer-product views of matrix multiplication
- [04_2d_register_tiling/04_matmul_2d_register_tiling_visualization.html](04_2d_register_tiling/04_matmul_2d_register_tiling_visualization.html): 2D register tiling
