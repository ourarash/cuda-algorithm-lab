/*
 * transpose_harness.cuh: shared host-side driver for the transpose ladder.
 *
 * Every step in matrix_transpose/ writes out[c][r] = in[r][c] for a row-major
 * rows x cols float matrix (step 00 is a plain copy, the speed limit the
 * others are measured against). A step provides one launcher; this header
 * owns the rest:
 * - random input, exact validation (a transpose only moves values);
 * - warmed-up timing reported as GB/s: every element is read once and written
 *   once, so bytes = 2 * rows * cols * 4, with % of peak DRAM bandwidth;
 * - --quick for a small size that is neither square nor a multiple of the
 *   tile size, and --rows / --cols to choose.
 *
 * All kernels use the classic configuration from NVIDIA's "An Efficient
 * Matrix Transpose in CUDA C/C++": 32 x 32 tiles handled by 32 x 8 threads,
 * so each thread moves TILE_DIM / BLOCK_ROWS = 4 elements.
 */
#pragma once

#include <cstdio>
#include <vector>

#include "lab.cuh"

constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;

using TransposeLauncher = void (*)(const float *in, float *out, int rows,
                                   int cols);

// Grid of TILE_DIM x TILE_DIM tiles over the input, BLOCK_ROWS x TILE_DIM
// threads per block.
inline dim3 transpose_grid(int rows, int cols) {
  return dim3(lab::ceil_div(cols, TILE_DIM), lab::ceil_div(rows, TILE_DIM));
}
inline dim3 transpose_block() { return dim3(TILE_DIM, BLOCK_ROWS); }

inline int run_transpose(const char *name, int argc, char **argv,
                         TransposeLauncher launch, bool is_copy = false) {
  lab::Args args(argc, argv);
  const int rows = static_cast<int>(args.get_int("rows", args.quick() ? 1000 : 8192));
  const int cols = static_cast<int>(args.get_int("cols", args.quick() ? 777 : 8192));
  const size_t count = static_cast<size_t>(rows) * cols;

  lab::print_device();
  std::printf("%s: %d x %d floats\n", name, rows, cols);

  const std::vector<float> input = lab::random_uniform<float>(count, -1.f, 1.f, 3);
  std::vector<float> expected(count);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      const size_t src = static_cast<size_t>(r) * cols + c;
      const size_t dst = is_copy ? src : static_cast<size_t>(c) * rows + r;
      expected[dst] = input[src];
    }
  }

  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, input.data(), count * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out, 0, count * sizeof(float)));

  launch(d_in, d_out, rows, cols);
  CUDA_CHECK_LAUNCH();
  std::vector<float> got(count);
  CUDA_CHECK(cudaMemcpy(got.data(), d_out, count * sizeof(float),
                        cudaMemcpyDeviceToHost));
  const bool pass = lab::check_equal(is_copy ? "copy" : "transpose", got, expected);

  const float ms = lab::time_ms([&] { launch(d_in, d_out, rows, cols); });
  lab::report(name, ms, 0, 2.0 * count * sizeof(float));

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return lab::finish(pass);
}
