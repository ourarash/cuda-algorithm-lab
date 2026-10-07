# CUDA Matrix Multiplication (GEMM) Evolution

This folder contains a sequential evolution of Matrix Multiplication (GEMM) implementations in CUDA. Each file represents a stepping stone in optimizing CUDA kernels, starting from a basic textbook implementation to a hardware-accelerated version.

Every stage computes `C = alpha * A @ B + beta * C` for row-major matrices and
shares one host-side driver, [gemm_harness.cuh](gemm_harness.cuh), so each
`.cu` file contains only its kernel and launch configuration. The harness
validates the result against a double-precision CPU reference (with
`beta != 0`, so the read-modify-write of `C` is tested too), then reports
GFLOP/s from warmed-up, repeated runs. Pass `--quick` for small sizes that are
not multiples of the tile sizes, or `--m`, `--n`, `--k` to choose your own.
For a vendor baseline, run `libraries/00_cublas_gemm`.

The progression follows the structure of Simon Boehm's article
[How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM),
which is an excellent companion read.

## Files Overview

### 0. `00_uncoalesced/00_matmul_uncoalesced.cu`

**The Anti-Pattern: Uncoalesced Memory Access**
This is the most basic implementation, mapping threads in a way that is antithetical to GPU architecture.

- **Characteristics:** The fastest-changing thread index (`threadIdx.x`) is mapped to matrix rows. Because memory is stored row-major, adjacent threads access memory locations that are far apart, leading to a catastrophic loss in memory bandwidth.

### 1. `01_coalesced/01_matmul_coalesced.cu`

**The First Fix: Coalesced Memory Access**
This kernel fixes the major flaw in the previous version with a simple one-line change to the thread-to-data mapping.

- **Characteristics:** The fastest-changing thread index (`threadIdx.x`) is now mapped to matrix columns. Adjacent threads now access adjacent memory locations, allowing the GPU to coalesce these reads into a single, efficient transaction.

### 2. `02_shared_memory/02_matmul_shared_memory.cu`

**Tiling via Shared Memory**
This version introduces **Tiling** to reduce redundant global memory reads.

- **Characteristics:** Threads cooperatively load a small "tile" of matrices `A` and `B` from global memory into fast on-chip **Shared Memory**, and every loaded element is reused `TILE_SIZE` times.
- **Why there is no padding:** Bank conflicts only happen between threads of one warp within one instruction. At a fixed `k`, a warp reads `tileA[ty][k]` (one address, a broadcast) and `tileB[k][tx]` (32 consecutive words, 32 different banks), so neither tile needs padding. Padding matters when a warp reads down a column at once; see `matrix_transpose/`.

### 3. `03_register_tiling/03_matmul_register_tiling.cu`

**Work-per-thread via Register Tiling**
This version increases arithmetic intensity by having each thread compute more than one output element.

- **Characteristics:** Each thread computes an 8x1 column of the output C-tile. It loads a value from shared memory into a private register and reuses that value 8 times.

### 4. `03_register_tiling/04_matmul_2d_register_tiling.cu`

**2D Register Tiling**
Builds upon 1D tiling by having each thread compute an 8x8 block of output elements.

- **Characteristics:** This massively boosts arithmetic intensity. A single thread loads 8 values of `A` and 8 values of `B` into its local registers, then executes 64 multiply-accumulates before touching shared memory again.
- **What still limits it:** every shared-memory load is a separate 32-bit instruction, and the reads have 2-way (A) and 4-way (B) bank conflicts. The file explains where they come from.

### 5. `04_vectorized/04_matmul_vectorized.cu`

**Vectorized Memory Access (`float4`) on top of 2D Register Tiling**
Keeps the 128x128 block tile and 8x8 outputs per thread from the previous stage and widens its memory instructions to 128 bits.

- **Characteristics:** Each thread moves one `float4` of `A` and one of `B` from global to shared memory per K tile. The `A` tile is stored transposed in shared memory, so each thread's 8 `A` values are contiguous and can be read with two `float4` loads, just like its 8 `B` values. `C` is read and written with `float4` too.
- **Requirement:** `N` and `K` must be multiples of 4 so every `float4` is 16-byte aligned.

### 6. `05_tensor_cores/05_matmul_tensor_cores.cu`

**Hardware Acceleration (WMMA API / Tensor Cores)**
This version uses NVIDIA's specialized **Tensor Cores**.

- **Characteristics:** It uses the Warp Matrix Multiply-Accumulate (WMMA) API. Each *warp* (32 threads) cooperatively computes one 16x16 tile of `C`; a block holds 4 warps arranged 2x2.
- **Note:** Tensor Cores here operate on mixed precision (FP16 inputs, FP32 accumulation). WMMA needs compute capability 7.0 or newer, and `M`, `N`, and `K` must be multiples of 16. This is the simplest correct WMMA kernel; it reads fragments straight from global memory, and the [roadmap](../roadmap.md) adds shared-memory staging, `mma.sync`, and Hopper's TMA + WGMMA.
