# CUDA Algorithm Lab

Lightweight CUDA examples for learning how GPU algorithms evolve from simple
versions to better ones.

This repo is organized as a teaching lab, not just a dump of kernels. Most
folders are arranged as small progressions, and several topics include HTML
visualizations to make the algorithm flow easier to follow.

## ✨ What You'll Find

- Progressive CUDA examples with clearer naming and folder structure
- Topics like reduction, scan, matrix multiplication, sparse ops, sorting, and
  warp-level programming
- Interactive visualizations for selected algorithms
- Shorter, more readable CUDA files with top-of-file intent and algorithm
  summaries

## 🗂️ Repo Layout

- `basics/`: CUDA basics, thread hierarchy, vector add, runtime API examples
- `libraries/`: library-based examples such as cuBLAS GEMM
- `matmul/`: progressively better GEMM kernels and visual explanations
- `matrix_transpose/`: shared-memory matrix transpose with bank-conflict avoidance
- `matmul_siboehm/`: study notes and implementations inspired by GEMM
  optimization walkthroughs
- `memory/`: memory-management focused examples
- `optimization/`: larger optimization-oriented experiments
- `reduction/`: reduction kernels plus visualizations
- `scan/`: inclusive and exclusive scan algorithms, from simple to multi-block
- `sort/`: sorting examples
- `sparse/`: sparse matrix-vector and sparse matrix-matrix examples
- `warp/`: warp shuffle and warp-level programming examples
- `xor/`: set symmetric difference example

## 🚀 Building

Build everything from the repo root:

```bash
make
```

Build a single topic:

```bash
make -C scan
make -C matmul
make -C reduction
```

Build one specific example folder:

```bash
make -C scan/00_kogge_stone
make -C matmul/02_shared_memory
make -C matrix_transpose
```

Clean generated binaries:

```bash
make clean
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

Some folders include HTML files that explain the algorithm step by step. A good
place to start:

- `reduction/00_naive/naive_reduction_visualization_tree.html`
- `reduction/01_shared/shared_reduction_visualization.html`
- `scan/00_kogge_stone/00_scan_kogge_stone_visualization.html`
- `matmul/02_shared_memory/02_matmul_shared_memory_visualization.html`

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
