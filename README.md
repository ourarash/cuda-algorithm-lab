# CUDA Algorithm Lab

Lightweight CUDA examples for learning how GPU algorithms evolve from simple
versions to better ones.

This repo is organized as a teaching lab, not just a dump of kernels. Most
folders are arranged as small progressions, and several topics include HTML
visualizations to make the algorithm flow easier to follow.

## ✨ What You'll Find

- Progressive CUDA examples with clear naming and folder structure
- Topics like reduction, scan, matrix multiplication, sparse ops, sorting, and
  warp-level programming
- Interactive visualizations for selected algorithms
- Short, readable CUDA files with top-of-file intent and algorithm summaries
- Every example checks its own result and reports a meaningful performance
  number (GFLOP/s or GB/s, with % of peak memory bandwidth where it applies)

## 🗂️ Repo Layout

- `basics/`: CUDA basics, thread hierarchy, vector add, runtime API examples
- `memory/`: memory-management focused examples
- `warp/`: warp shuffle and warp-level programming examples
- `reduction/`: reduction kernels plus visualizations
- `scan/`: inclusive and exclusive scan algorithms, from simple to multi-block
- `matmul/`: progressively better GEMM kernels and visual explanations
- `matrix_transpose/`: shared-memory matrix transpose with bank-conflict avoidance
- `libraries/`: library-based examples such as cuBLAS GEMM
- `sort/`: sorting examples
- `sparse/`: sparse matrix-vector (COO, CSR, ELL) and sparse matrix-matrix examples
- `xor/`: set symmetric difference with Thrust
- `optimization/`: larger optimization-oriented experiments
- `common/`: `lab.cuh`, the small shared header for error checking, timing,
  validation, and reporting

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

Test and sanitize everything:

```bash
make test       # run every example at full size; each must PASS
make sanitize   # run every example under compute-sanitizer memcheck and racecheck
make clean      # delete build/
```

`make test` and `make sanitize` are thin wrappers around `ctest`, so you can
also select examples directly, for example
`ctest --test-dir build -L run -R matmul`.

To compile a single file by hand, add the shared header to the include path:

```bash
nvcc -O3 -I common basics/02_vector_add/02_vector_add.cu -o vector_add
```

## 🧠 Recommended Path

If you're using this repo to learn, a good order is:

1. `basics/`
2. `reduction/`
3. `scan/`
4. `matmul/`
5. `warp/`
6. `sparse/`

## 🌐 Visualizations

Some folders include HTML files that explain the algorithm step by step. Open
them directly in a browser. A good place to start:

- `reduction/00_naive/naive_reduction_visualization_tree.html`
- `reduction/01_shared/shared_reduction_visualization.html`
- `scan/00_kogge_stone/00_scan_kogge_stone_visualization.html`
- `matmul/02_shared_memory/02_matmul_shared_memory_visualization.html`
- `matrix_transpose/00_transpose_visualization.html`

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
