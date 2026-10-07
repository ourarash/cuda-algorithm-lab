# Roadmap

This document tracks the plan for turning CUDA Algorithm Lab into a
comprehensive, trustworthy resource for learning CUDA and kernel optimization.
It started from a full review of the repository (October 2026). Items are
grouped into three phases; check them off as they land.

The guiding principle: **every example must build, validate itself, exit
non-zero on failure, and report a performance number that means something**
(GFLOP/s, GB/s, % of peak, or % of a vendor library).

---

## Review findings

### Builds that fail

| Where | Problem |
|---|---|
| `matmul/04_vectorized/Makefile`, `matmul/05_tensor_cores/Makefile` | Targets reference `05_*.cu` / `06_*.cu`, which no longer exist after a rename, so the root `make` fails. |
| `matmul/05_tensor_cores/Makefile` | `-arch=sm_70`: CUDA 13 removed offline compilation for Maxwell, Pascal, and Volta. The comment also calls sm_70 "Turing" (it is Volta). |
| `basics/03_runtime_api_device_query` | Uses `cudaDeviceProp::clockRate` and `memoryClockRate`, which were removed in CUDA 13. Use `cudaDeviceGetAttribute`. |
| `sparse/SpMV_EllPack.cu` | References an undefined `CSRMatrix` and members that do not exist. The ELL conversion is also wrong: it scans the first `maxNonZero` dense columns instead of the row's nonzeros. |
| `optimization/00_ant_colony_tsp` | Calls `atomicMin` on a `float`; CUDA C++ only provides integer overloads. |
| `reduction/` | No Makefiles and not listed in the root `SUBDIRS`, although the README documents `make -C reduction`. |

### Wrong results

| Where | Problem |
|---|---|
| `matmul/04_vectorized` | A 32×32 block writes into `float4 tileB[32][8]`, so threads with `x ≥ 8` write out of bounds in shared memory. The grid only reaches column ~351 of 1024. |
| `matmul/05_tensor_cores` | Each block launches 8 warps that all compute the same 16×16 tile: 8× redundant work and a read-modify-write race on C whenever β ≠ 0. |
| `matmul/02_shared_memory`, `matmul/04_vectorized` | The tile loop runs `K / TILE_SIZE` times, silently dropping the K tail even though the loads are bounds-checked. |
| `sort/01_merge_sort` | An unpaired trailing chunk is never copied to the destination buffer before the ping-pong swap, so data is lost for any N whose chunk count is not a power of two. |
| `optimization/00_ant_colony_tsp` | The pheromone kernel launches `dim3(100, 100)` = 10,000 threads per block, above the 1,024 limit, so it fails silently and pheromones never update. The best-path copy is also racy. |
| `libraries/00_cublas_gemm` | Inputs are written row-major but cuBLAS reads column-major, so the program prints Aᵀ·Bᵀ instead of A·B. |
| `scan/00`–`scan/04` | An absolute tolerance of 1e-4 is below one float ULP at 1024 (1.2e-4). A float32 simulation of the GPU summation order fails on 200 of 200 random inputs, with errors up to 1.5e-3. |
| `reduction/00_naive` | Ignores `N`, so it reads out of bounds for any N that is not a multiple of 512. |

### Wrong teaching

- **Padding misconception** in `matmul/02_shared_memory` (code comment,
  visualization, and `matmul/README.md`) and in `matmul_siboehm/*_pad.cu`. A
  warp reads `tileB[k][threadIdx.x]` along a row, which is already
  conflict-free, so the padding does nothing. Matrix transpose is the case
  where padding matters, and that example gets it right. The 2D register-tiling
  kernel does have real conflicts on `Bs[dotIdx][threadCol * TN + j]`; the fix
  there is a different shared-memory layout, not padding.
- **The matmul ladder goes backwards.** The vectorized stage (1×4 outputs per
  thread) came after 2D register tiling (8×8 outputs per thread), so it would
  be slower than the step before it.
- `matmul/04_vectorized/cuda-matrix-memory-mapping.html` visualizes naive 2D
  thread indexing, not `float4` access. It belongs with the naive kernels.

### Security

Four matmul visualizations loaded `https://polyfill.io/...`. That domain was
taken over in 2024 and served malicious JavaScript. Modern browsers do not need
the polyfill.

### Problems shared by every example

- Almost every program returns 0 even when validation fails, so automation
  cannot catch regressions.
- Timing is a single cold launch with no warmup, which includes lazy module
  loading.
- Results are reported in milliseconds instead of GFLOP/s or GB/s.
- The matmul tests use integer-valued inputs with β = 0, which hides precision
  problems and the tensor-core race.

---

## Phase 1: correctness and infrastructure

Status: code changes are in. Every example type-checks against the CUDA 12.8
headers (host and device code); the first `nvcc` build runs in CI, and the
examples still need a first run on a GPU (`make test`, `make sanitize`).

- [x] Fix every build failure listed above.
- [x] Fix every wrong-result bug listed above.
- [x] Fix the padding explanation (code, README, visualization).
- [ ] Move the misplaced visualization to `matmul/01_coalesced/`.
- [x] Rebuild the vectorized matmul stage on top of 2D register tiling so the
      ladder improves monotonically.
- [x] Remove the `polyfill.io` script tags.
- [x] Add a shared `common/` header: error checking, warmup + repeated timing
      (median), GFLOP/s and GB/s with % of peak bandwidth, validation against a
      higher-precision reference with relative tolerance, non-zero exit code on
      failure, and a `--quick` flag for small problem sizes.
- [x] Switch the build to CMake with `ctest`; keep a thin root `Makefile` so
      `make`, `make test`, and `make sanitize` still work.
- [ ] Delete the old per-folder Makefiles, which the CMake build replaces.
- [x] Run `compute-sanitizer` (memcheck and racecheck) through `ctest` labels.
- [x] Credit Simon Boehm's article in `matmul/README.md`.
- [ ] Delete `matmul_siboehm/` (a duplicate of `matmul/`) and the superseded
      sparse files (`sparse/SpMV_CSR.cu`, `sparse/SpMV_EllPack.cu`,
      `sparse/01_spgemm_cusparse/`, now `sparse/01_spmv_csr/`,
      `sparse/02_spmv_ell/`, and `sparse/03_spgemm_cusparse/`).
- [x] Add a compile-only GitHub Actions workflow (hosted runners have no GPU).
- [ ] Run `make test` and `make sanitize` on a GPU and fix anything they find.

## Phase 2: make the optimization story measurable

- [ ] Rebuild the GEMM ladder so each step is faster than the last:
  1. naive
  2. coalesced
  3. shared memory
  4. 1D register tiling
  5. 2D register tiling
  6. vectorized loads + transposed A tile
  7. warptiling
  8. double buffering with `cp.async`
  9. WMMA with shared-memory staging
  10. `mma.sync` + `ldmatrix` + swizzled shared memory
  11. Hopper TMA + WGMMA (`sm_90a`)

  Optionally follow with Blackwell `tcgen05` or a CuTe version. Print cuBLAS as
  the baseline on every run.
- [ ] Full reduction ladder: the Harris steps (interleaved → sequential →
      first add during load → unroll last warp → complete unroll →
      grid-stride), then warp shuffle, cooperative groups, single-pass with
      atomics or last-block, `float4` loads, and a CUB comparison.
- [ ] Transpose ladder: copy baseline → naive → shared memory → padded →
      swizzled, all in GB/s.
- [ ] Per-topic README results tables: GPU, time, GFLOP/s or GB/s, % of
      cuBLAS or % of peak, plus the one Nsight Compute metric that proves each
      step's claim (for example
      `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` for bank
      conflicts, sectors per request for coalescing).
- [ ] Roofline plot for the GEMM ladder.

### Cleanup

- [ ] Remove duplicate visualizations (two single-thread register-tiling views,
      two naive-reduction views); keep the best of each.
- [ ] Turn `test-viz.js` into a real visualization smoke test with a
      `package.json`, or delete it.
- [ ] Move `plugins/` and `.agents/` (personal Codex tooling) out of the public
      tree.
- [ ] Move `matmul/matmul_presentation.pptx` and `generate_pptx.py` to `docs/`
      or a release asset.
- [ ] Move `xor/`, the epsilon sorts, and ant colony optimization to
      `applications/` or `libraries/thrust/`; they are library demos or niche
      algorithms rather than CUDA fundamentals.

## Phase 3: new content

- [ ] Scan: warp-shuffle scan, single-pass decoupled look-back (the CUB
      algorithm), CUB comparison.
- [ ] Core patterns: histogram (privatization, aggregation), stream
      compaction, radix sort built from the repo's own scan and histogram,
      bitonic sort (also replacing the single-thread insertion sort in merge
      sort), stencil and convolution (constant memory, halo tiles).
- [ ] Modern ML kernels: online softmax, LayerNorm/RMSNorm, a minimal
      FlashAttention forward pass, and a PyTorch C++/CUDA extension wrapping
      one of them.
- [ ] Memory and concurrency: pinned vs. pageable bandwidth, streams with
      copy/compute overlap, CUDA Graphs, unified memory with prefetch and
      advise.
- [ ] Debugging chapter: intentionally buggy kernels (race, out-of-bounds,
      missing sync, divergent `__syncthreads`) to diagnose with
      `compute-sanitizer`.
- [ ] Sparse: CSR scalar vs. CSR vector (warp per row), ELL, hybrid, cuSPARSE
      SpMV, and a Matrix Market loader so realistic matrices expose load
      imbalance.
- [ ] Basics: occupancy and `__launch_bounds__`, a warp-divergence demo,
      cooperative groups.
