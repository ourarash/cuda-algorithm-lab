# CUDA Algorithm Lab

[![build](https://github.com/ourarash/cuda-algorithm-lab/actions/workflows/build.yml/badge.svg)](https://github.com/ourarash/cuda-algorithm-lab/actions/workflows/build.yml)
[![CUDA 12.8 | 13.3](https://img.shields.io/badge/CUDA-12.8%20%7C%2013.3-76B900?logo=nvidia&logoColor=white)](https://github.com/ourarash/cuda-algorithm-lab/actions/workflows/build.yml)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)](https://en.cppreference.com/w/cpp/17)
[![CMake 3.24+](https://img.shields.io/badge/CMake-3.24%2B-064F8C?logo=cmake&logoColor=white)](CMakeLists.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Lightweight CUDA examples for learning how GPU algorithms evolve from simple
versions to better ones.

This repo is organized as a teaching lab, not just a dump of kernels. Most
folders are arranged as small progressions, and several topics include HTML
visualizations to make the algorithm flow easier to follow.

## ✨ What You'll Find

- Three optimization ladders where every step changes one idea and reports
  the number that proves it:
  - [matmul/](matmul/): 12 GEMM steps, from a naive kernel through register
    tiling, warptiling, and `cp.async` double buffering to Tensor Cores with
    WMMA, raw `mma.sync` + `ldmatrix`, and Hopper TMA + WGMMA, each compared
    with cuBLAS
  - [reduction/](reduction/): 13 steps, Harris's classic sequence updated for
    modern GPUs, then warp shuffles, cooperative groups, single-pass, `float4`,
    and CUB
  - [matrix_transpose/](matrix_transpose/): copy baseline, naive, shared
    memory, padding, and swizzling
- Modern ML kernels in [ml/](ml/): softmax (three-pass and online),
  LayerNorm, RMSNorm, naive attention vs. FlashAttention, and a PyTorch
  extension
- Parallel building blocks: scan (up to single-pass decoupled look-back),
  histogram, stream compaction, and a radix sort built from them
- Convolution and stencils with halo tiles, memory and concurrency (pinned
  memory, streams, CUDA Graphs, unified memory), sparse formats on irregular
  matrices, and a debugging chapter built around compute-sanitizer
- Interactive visualizations for selected algorithms
- Short, readable CUDA files with top-of-file intent and algorithm summaries
- Every example checks its own result and reports a meaningful performance
  number (GFLOP/s or GB/s, with % of peak memory bandwidth where it applies)

## 🗂️ Repo Layout

- `basics/`: thread hierarchy, vector add, device query, occupancy and
  `__launch_bounds__`, warp divergence, cooperative groups
- `memory/`: pinned memory, streams, CUDA Graphs, unified memory
  ([README](memory/README.md))
- `warp/`: warp shuffle and warp-level programming examples
- `reduction/`: the reduction ladder ([README](reduction/README.md))
- `scan/`: prefix sums, from single-block textbook scans to single-pass
  decoupled look-back ([README](scan/README.md))
- `histogram/`: atomics, privatization, aggregation ([README](histogram/README.md))
- `compaction/`: stream compaction, stable and unstable ([README](compaction/README.md))
- `matmul/`: the GEMM ladder ([README](matmul/README.md))
- `matrix_transpose/`: the transpose ladder ([README](matrix_transpose/README.md))
- `sort/`: counting, merge, bitonic, and radix sort ([README](sort/README.md))
- `convolution/` and `stencil/`: 2D convolution and a 3D stencil with halo
  tiles ([README](convolution/README.md), [README](stencil/README.md))
- `ml/`: softmax, LayerNorm, RMSNorm, attention, FlashAttention, and a
  PyTorch extension ([README](ml/README.md))
- `debugging/`: deliberate bugs to find with compute-sanitizer
  ([README](debugging/README.md))
- `sparse/`: COO, CSR (scalar and vector), ELL, hybrid, and cuSPARSE, on
  uniform and power-law matrices, plus Matrix Market loading
  ([README](sparse/README.md))
- `libraries/`: cuBLAS GEMM and a Thrust set-operations example
- `applications/`: larger examples (ant colony optimization for the TSP,
  approximate "epsilon" sorting)
- `common/`: `lab.cuh`, the small shared header for error checking, timing,
  validation, and reporting
- `tools/`: benchmark tables, roofline plot, visualization checks
- `docs/`: a slide deck on the GEMM ladder and its generator

See [roadmap.md](roadmap.md) for what is planned next.

## 🧰 Requirements

- An NVIDIA GPU and the CUDA Toolkit, 12.x or 13.x. CUDA 13 supports compute
  capability 7.5 (Turing) and newer.
- CMake 3.24 or newer.
- `compute-sanitizer` (ships with the CUDA Toolkit) for `make sanitize`.

## 🚀 Building and Running

Build everything from the repo root:

```bash
make
```

This configures CMake in `build/` for the GPU in your machine and compiles
every example into `build/bin/<topic>/<name>`. To build for other GPUs, pass
the architectures explicitly:

```bash
make CMAKE_ARGS='-DCMAKE_CUDA_ARCHITECTURES=80;90'
```

Run a single example:

```bash
./build/bin/matmul/02_matmul_shared_memory
./build/bin/matmul/02_matmul_shared_memory --m 2048 --n 2048 --k 2048
./build/bin/scan/06_scan_multiblock_gpu_fixup --quick
```

Every example prints its GPU, problem size, a validation line, and timing,
and finishes with `PASS` or `FAIL`. The exit code is non-zero on failure.
`--quick` switches to small, deliberately awkward sizes (for example, matrix
dimensions that are not multiples of the tile size).

Test, measure, and sanitize everything:

```bash
make test       # run every example at full size; each must PASS
make sanitize   # run every example under compute-sanitizer memcheck and racecheck
make bench      # results tables for every topic that reports performance
make roofline   # roofline plot of the GEMM ladder (matplotlib; uses ncu if installed)
make check-viz  # smoke-test the HTML visualizations (Node.js)
make clean      # delete build/
```

Some examples need a particular GPU (for example, the Hopper GEMM needs
compute capability 9.0). On other GPUs they print `SKIP` and exit with code 77,
which `ctest` reports as skipped rather than failed.

`make test` and `make sanitize` are thin wrappers around `ctest`, so you can
also select examples directly, for example
`ctest --test-dir build -L run -R matmul`.

To compile a single file by hand, add the shared header to the include path:

```bash
nvcc -O3 -I common basics/02_vector_add/02_vector_add.cu -o vector_add
```

## 🧠 Recommended Path

If you're using this repo to learn, a good order is:

1. `basics/` and `memory/`
2. `reduction/`
3. `matrix_transpose/`
4. `scan/`, then `histogram/` and `compaction/`
5. `sort/` (radix sort ties the previous three together)
6. `convolution/` and `stencil/`
7. `matmul/`
8. `ml/`
9. `sparse/`
10. `debugging/`, any time something goes wrong

## 🌐 Visualizations

Some folders include HTML files that explain the algorithm step by step. Open
them directly in a browser. A good place to start:

- `reduction/00_naive/naive_reduction_visualization_tree.html`
- `reduction/03_sequential_addressing/shared_reduction_visualization.html`
- `scan/00_kogge_stone/00_scan_kogge_stone_visualization.html`
- `matmul/02_shared_memory/02_matmul_shared_memory_visualization.html`
- `matrix_transpose/03_padded/03_transpose_bank_conflicts_visualization.html`

The topic READMEs list the rest.

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
