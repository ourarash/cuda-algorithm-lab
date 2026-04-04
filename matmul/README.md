# CUDA Matrix Multiplication (GEMM) Evolution

This folder contains a sequential evolution of Matrix Multiplication (GEMM) implementations in CUDA. Each file represents a stepping stone in optimizing CUDA kernels, starting from a basic textbook implementation to a highly hardware-optimized version.

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

- **Characteristics:** Threads cooperatively load a small "tile" of matrices `A` and `B` from global memory into ultra-fast **Shared Memory**.
- **Bonus Trick:** It includes a `[TILE_SIZE][TILE_SIZE+1]` array padding. This slight offset prevents "Shared Memory Bank Conflicts".

### 3. `03_register_tiling/03_matmul_register_tiling.cu`

**Work-per-thread via Register Tiling**
This version increases arithmetic intensity by having each thread compute more than one output element.

- **Characteristics:** Each thread computes an 8x1 column of the output C-tile. It loads a value from shared memory into a private register and reuses that value 8 times.

### 4. `03_register_tiling/04_matmul_2d_register_tiling.cu`

**2D Register Tiling**
Builds upon 1D tiling by having each thread compute an 8x8 block of output elements.

- **Characteristics:** This massively boosts arithmetic intensity. A single thread loads 8 values of `A` and 8 values of `B` into its local registers, then executes 64 math operations without needing to fetch from shared memory again.

### 5. `04_vectorized/04_matmul_vectorized.cu`

**Vectorized Memory Access (`float4`)**
Builds upon the shared memory implementation by widening the memory fetches.

- **Characteristics:** By using CUDA's built-in `float4` vector type, each thread fetches 128 bits (four 32-bit floats) at once in a single instruction.

### 6. `05_tensor_cores/05_matmul_tensor_cores.cu`

**Hardware Acceleration (WMMA API / Tensor Cores)**
This version uses NVIDIA's specialized **Tensor Cores**.

- **Characteristics:** It uses the Warp Matrix Multiply-Accumulate (WMMA) API. An entire *Warp* (32 threads) is programmed to cooperatively execute matrix math at the hardware level.
- **Note:** Tensor Cores operate on mixed precision (FP16 inputs, FP32 accumulation).
