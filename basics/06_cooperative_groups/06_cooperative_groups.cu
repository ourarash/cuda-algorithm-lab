/*
 * Cooperative Groups and Grid-Wide Synchronization
 *
 * Intention:
 * Cooperative Groups is the CUDA API for naming sets of threads and
 * synchronizing them: a tile of a warp, a warp, a block, or the whole grid.
 * __syncthreads() can only synchronize one block, so algorithms with a
 * global dependency ("first compute a total over all the data, then use it
 * everywhere") normally need two kernel launches. A cooperative launch
 * guarantees that all blocks are resident at once, which makes grid.sync()
 * possible inside one kernel.
 *
 * High-Level Algorithm (normalize a vector to sum 1 in one kernel):
 * - Phase 1: each block sums its grid-stride share; thread_block_tile<32>
 *   (a warp) reduces with cg::reduce, and tile leaders combine into the
 *   block's partial sum, written to partials[block].
 * - grid.sync(): every block waits until every block has written its sum.
 * - Phase 2: each block adds up all partial sums (few, so cheap) and divides
 *   its elements by the total.
 *
 * Requirements: the device must support cooperative launch, the kernel must
 * be launched with cudaLaunchCooperativeKernel, and the grid may not exceed
 * the number of blocks that can be resident at once (queried with the
 * occupancy API below).
 */
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <vector>

#include "lab.cuh"

namespace cg = cooperative_groups;

constexpr int THREADS = 256;

__global__ void normalize(float *x, float *partials, int n) {
  cg::grid_group grid = cg::this_grid();
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
  __shared__ float warp_sums[THREADS / 32];

  // Phase 1: this block's partial sum.
  float sum = 0.0f;
  for (int i = grid.thread_rank(); i < n; i += grid.size()) sum += x[i];
  sum = cg::reduce(warp, sum, cg::plus<float>());
  if (warp.thread_rank() == 0) warp_sums[warp.meta_group_rank()] = sum;
  block.sync();
  if (block.thread_rank() == 0) {
    float block_sum = 0.0f;
    for (int w = 0; w < THREADS / 32; ++w) block_sum += warp_sums[w];
    partials[blockIdx.x] = block_sum;
  }

  grid.sync();  // All partial sums are now written and visible.

  // Phase 2: total, then normalize.
  float total = 0.0f;
  for (int b = 0; b < gridDim.x; ++b) total += partials[b];
  const float inv = 1.0f / total;
  for (int i = grid.thread_rank(); i < n; i += grid.size()) x[i] *= inv;
}

int main(int argc, char **argv) {
  lab::Args args(argc, argv);
  const int n = static_cast<int>(args.get_int("n", args.quick() ? 100003 : 1 << 24));

  lab::print_device();
  int device = 0, supported = 0, sms = 0, blocks_per_sm = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaDeviceGetAttribute(&supported, cudaDevAttrCooperativeLaunch, device));
  if (!supported) {
    return lab::skip("this device does not support cooperative launches");
  }
  CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, normalize, THREADS, 0));
  int blocks = sms * blocks_per_sm;  // Every block must be resident at once
  blocks = std::min(blocks, lab::ceil_div(n, THREADS));
  printf("Cooperative launch: %d blocks (%d per SM x %d SMs max)\n", blocks,
         blocks_per_sm, sms);

  const std::vector<float> h_x = lab::random_uniform<float>(n, 0.f, 1.f, 101);
  double total = 0.0;
  for (float v : h_x) total += v;
  std::vector<double> expected(n);
  for (int i = 0; i < n; ++i) expected[i] = h_x[i] / total;

  float *d_x, *d_partials;
  CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_partials, blocks * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  int n_arg = n;
  void *kernel_args[] = {&d_x, &d_partials, &n_arg};
  CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void *>(normalize), blocks,
                                         THREADS, kernel_args, 0, 0));
  CUDA_CHECK_LAUNCH();
  std::vector<float> got(n);
  CUDA_CHECK(cudaMemcpy(got.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost));
  const bool pass = lab::check_close("normalized vector", got, expected, 1e-4, 0.0);

  // Timing re-normalizes the already-normalized data, which costs the same.
  const float ms = lab::time_ms([&] {
    CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void *>(normalize), blocks,
                                           THREADS, kernel_args, 0, 0));
  });
  lab::report("normalize (one cooperative kernel)", ms, 0, 3.0 * n * sizeof(float));

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_partials));
  return lab::finish(pass);
}
