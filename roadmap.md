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

Status: done, except for the first run on a GPU. CI compiles every example
with CUDA 12.8 and 13.3; the examples have not yet been executed on a GPU.

- [x] Fix every build failure listed above.
- [x] Fix every wrong-result bug listed above.
- [x] Fix the padding explanation (code, README, visualization).
- [x] Move the misplaced visualization to `matmul/01_coalesced/`.
- [x] Rebuild the vectorized matmul stage on top of 2D register tiling so the
      ladder improves monotonically.
- [x] Remove the `polyfill.io` script tags.
- [x] Add a shared `common/` header: error checking, warmup + repeated timing
      (median), GFLOP/s and GB/s with % of peak bandwidth, validation against a
      higher-precision reference with relative tolerance, non-zero exit code on
      failure, and a `--quick` flag for small problem sizes.
- [x] Switch the build to CMake with `ctest`; keep a thin root `Makefile` so
      `make`, `make test`, and `make sanitize` still work.
- [x] Delete the old per-folder Makefiles, which the CMake build replaces.
- [x] Run `compute-sanitizer` (memcheck and racecheck) through `ctest` labels.
- [x] Credit Simon Boehm's article in `matmul/README.md`.
- [x] Delete `matmul_siboehm/` (a duplicate of `matmul/`) and the superseded
      sparse files, now `sparse/01_spmv_csr/`, `sparse/02_spmv_ell/`, and
      `sparse/03_spgemm_cusparse/`.
- [x] Add a compile-only GitHub Actions workflow (hosted runners have no GPU).
- [ ] Run `make test` and `make sanitize` on a GPU and fix anything they find.

## Phase 2: make the optimization story measurable

Status: done, except for measurements, which need a GPU. Every new kernel was
type-checked locally and compiled by nvcc in CI. Their indexing (including the
`ldmatrix`/`mma.sync` fragment layouts, the swizzles, and the TMA/WGMMA shared
memory layout) was checked against a thread-by-thread NumPy emulation, but no
kernel has run on real hardware yet; every one validates itself when it does.

- [x] Rebuild the GEMM ladder (`matmul/00`-`11`), printing cuBLAS as the
      baseline on every run:
  1. naive (00) and coalesced (01)
  2. shared memory (02)
  3. 1D (03) and 2D (04) register tiling
  4. vectorized loads + transposed A tile (05)
  5. warptiling (06)
  6. double buffering with `cp.async` (07)
  7. WMMA, first without (08) and then with (09) shared-memory staging
  8. `mma.sync` + `ldmatrix` + `cp.async` + swizzled shared memory (10)
  9. Hopper TMA + WGMMA (`sm_90a`, 11)
- [x] Full reduction ladder (`reduction/00`-`12`): the Harris steps
      (interleaved divergent → interleaved with bank conflicts → sequential
      → first add during load → unroll last warp → complete unroll →
      grid-stride), then warp shuffle, cooperative groups, single-pass
      (last block), `float4` loads, and CUB.
- [x] Transpose ladder (`matrix_transpose/00`-`04`): copy baseline → naive →
      shared memory → padded → swizzled, all in GB/s.
- [x] Per-topic README tables naming the Nsight Compute metric that proves
      each step's claim, and `make bench` (`tools/bench.py`) to generate the
      results tables.
- [ ] Fill the results tables with measurements from a real GPU.
- [x] Roofline plot tool for the GEMM ladder (`make roofline`,
      `tools/roofline.py`; measures DRAM traffic with Nsight Compute when
      available).
- [ ] Commit a roofline plot measured on a real GPU.
- [ ] Optional: a Blackwell `tcgen05` GEMM, or a CuTe version.
- [ ] Optional: a multi-stage, warp-specialized Hopper GEMM with clusters and
      TMA multicast.

### Cleanup

- [x] Remove duplicate visualizations (kept the more complete single-thread
      register-tiling view and the tree view of the naive reduction).
- [x] Replace `test-viz.js` with `tools/check_visualizations.mjs` (run in CI
      and by `make check-viz`).
- [x] Move `plugins/` and `.agents/` (personal Codex tooling) out of the public
      tree (untracked and ignored; they stay on disk).
- [x] Move `matmul/matmul_presentation.pptx` and `generate_pptx.py` to `docs/`.
- [x] Move the Thrust set-operations example to `libraries/`, and ant colony
      optimization and the epsilon sorts to `applications/`.

## Phase 3: new content

Status: done, except for running the new examples on a GPU (the same open
item as Phases 1 and 2). Each new kernel was type-checked locally and
compiled by nvcc in CI; the trickier logic (decoupled look-back under random
block schedules, radix sort stability, bitonic networks, FlashAttention
tiling, halo loads, and the float tolerances of the ML kernels) was checked
with NumPy emulations.

- [x] Scan (`scan/07`-`09`): warp-shuffle reduce-then-scan, single-pass
      decoupled look-back, CUB comparison.
- [x] Core patterns:
  - [x] histogram (`histogram/`): global atomics, shared-memory
        privatization, aggregation with vector loads, CUB;
  - [x] stream compaction (`compaction/`): scan-based (stable),
        warp-aggregated atomics (unstable), CUB;
  - [x] radix sort built from the repo's own histogram and scan
        (`sort/03`), with CUB as the baseline (`sort/04`);
  - [x] bitonic sort (`sort/02`), also replacing the single-thread insertion
        sort in merge sort;
  - [x] convolution (`convolution/`: constant memory, halo tiles) and a 3D
        stencil (`stencil/`: register streaming along z).
- [x] Modern ML kernels (`ml/`): three-pass and online softmax, LayerNorm
      (Welford), RMSNorm, naive attention, a minimal FlashAttention forward
      pass, and a PyTorch C++/CUDA extension wrapping RMSNorm and softmax.
- [x] Memory and concurrency (`memory/01`-`04`): pinned vs. pageable
      bandwidth, streams with copy/compute overlap, CUDA Graphs, unified
      memory with prefetch.
- [x] Debugging chapter (`debugging/`): out-of-bounds (memcheck), shared
      memory race (racecheck), uninitialized memory (initcheck), wrong
      `__syncwarp` mask (synccheck), each wired into ctest as a test that the
      sanitizer catches the bug.
- [x] Sparse (`sparse/04`-`07`): CSR scalar vs. CSR vector on power-law
      matrices, hybrid ELL + COO, cuSPARSE SpMV, and a Matrix Market loader
      (`--mtx`).
- [x] Basics (`basics/04`-`06`): occupancy and `__launch_bounds__`, a
      warp-divergence demo, cooperative groups with grid-wide sync.
- [ ] Build and run the PyTorch extension test with PyTorch on a GPU (CI has
      no PyTorch, so it only checks that the test script parses).

## Ideas beyond the roadmap

- Causal masking, a backward pass, and Tensor Cores for FlashAttention.
- A warp-parallel look-back for the decoupled look-back scan, and 8-bit
  digits with shared-memory local sorting for the radix sort.
- Multi-GPU examples: peer-to-peer copies and NCCL all-reduce.
