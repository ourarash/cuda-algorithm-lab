/*
 * Naive Parallel Reduction
 *
 * High-Level Algorithm:
 * This kernel implements a basic tree-based parallel reduction directly in
 * global memory. The algorithm works by pairing up elements and continually
 * halving the number of elements to process in each step until a single sum remains.
 *
 * Tree Structure (Growing Stride):
 * - Step 1: Stride 1. Threads add elements 1 apart (e.g., T0 handles [0]+[1], T1 handles [2]+[3]).
 * - Step 2: Stride 2. Threads add elements 2 apart (e.g., T0 handles [0]+[2], T2 handles [4]+[6]).
 * - Step N: Stride doubles each iteration.
 *
 * Drawbacks:
 * - Uses slow global memory for all intermediate steps.
 * - High Warp Divergence: The condition `threadIdx.x % stride == 0` leaves many threads
 *   in a 32-thread warp inactive while a few do the work, wasting compute cycles.
 * - Uncoalesced Memory: As the stride grows, active threads access memory locations that
 *   are far apart, breaking memory coalescing rules and tanking memory bandwidth.
 *
 * Why not a shrinking stride here?
 * - We *could* use a shrinking stride (packed threads) in global memory to fix warp
 *   divergence. However, this naive implementation uses a growing stride because it represents
 *   the most direct conceptual translation of a binary tree. Even if we packed the threads,
 *   it would still suffer from massive global memory latency, which is why the next real
 *   optimization leap is to move to shared memory entirely.
 *
 * The block partial sums are added up on the host. The input size is
 * deliberately not a multiple of the 2 * BLOCK_SIZE elements each block
 * handles, so the bounds checks in the kernel are exercised.
 */
#include <cstdio>
#include <vector>

#include "lab.cuh"

#define BLOCK_SIZE 256  // Number of threads per block

// -------------------------------------------------------------------------
// Naive Parallel Reduction Kernel
// -------------------------------------------------------------------------
// This is a basic form of parallel reduction without employing shared memory.
// It works in-place directly on the global memory `input` array.
// Note: Each thread block processes 2 * BLOCK_SIZE elements.
//
// Drawbacks of this naive approach:
// 1. Heavy reliance on slow global memory.
// 2. High warp divergence (e.g., `if (threadIdx.x % stride == 0)` causes only
//    a fraction of threads in a warp to actually do work while others idle).
// 3. Uncoalesced memory accesses pattern in later iterations of the loop.
__global__ void reduction(float *input, float *partialSums, unsigned int N) {
  // Calculate global segment start for the current block.
  // Each block handles 2 * blockDim.x elements.
  unsigned int segment = blockIdx.x * blockDim.x * 2;

  // Calculate specific thread's starting index within the segment.
  // We multiply by 2 because each thread pair starts with a distance of 1 stride.
  unsigned int i = segment + threadIdx.x * 2;

  // Stride doubles each iteration for tree-based reduction: 1, 2, 4, 8, ...
  for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
    // Only active threads perform the addition. This causes high warp
    // divergence. In the last block, the right operand may lie past the end
    // of the array; it would contribute 0, so the addition is skipped.
    if (threadIdx.x % stride == 0 && i + stride < N) {
      input[i] += input[i + stride];
    }
    // Block-wide synchronization to ensure all threads finish a level of the tree
    // before moving to the next level.
    __syncthreads();
  }

  // The total sum for this block ends up in the first element processed by thread 0.
  // Write the partial sum from this block to global memory.
  if (threadIdx.x == 0) {
    partialSums[blockIdx.x] = input[i];
  }
}

int main(int argc, char **argv) {
  lab::Args args(argc, argv);
  const unsigned int N = static_cast<unsigned int>(
      args.get_int("n", args.quick() ? 100003 : (1 << 24) + 123));
  const unsigned int numBlocks = lab::ceil_div(N, 2u * BLOCK_SIZE);

  lab::print_device();
  printf("Naive reduction of %u floats\n", N);

  const std::vector<float> h_input = lab::random_uniform<float>(N, 0.f, 1.f, 7);
  float *d_input, *d_partialSums;
  CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_partialSums, numBlocks * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float),
                        cudaMemcpyHostToDevice));

  reduction<<<numBlocks, BLOCK_SIZE>>>(d_input, d_partialSums, N);
  CUDA_CHECK_LAUNCH();

  // Add the block partial sums on the host. Double precision keeps this
  // final step from adding error of its own.
  std::vector<float> h_partialSums(numBlocks);
  CUDA_CHECK(cudaMemcpy(h_partialSums.data(), d_partialSums,
                        numBlocks * sizeof(float), cudaMemcpyDeviceToHost));
  double gpu_sum = 0.0;
  for (float partial : h_partialSums) {
    gpu_sum += partial;
  }

  double cpu_sum = 0.0;
  for (float x : h_input) {
    cpu_sum += x;
  }
  printf("GPU sum: %.6f\nCPU sum: %.6f\n", gpu_sum, cpu_sum);
  // Each block's tree adds up at most 2 * BLOCK_SIZE floats in 9 levels, so
  // the relative error stays near 1e-7; a wrong index is off by far more.
  const std::vector<double> got = {gpu_sum}, expected = {cpu_sum};
  const bool pass = lab::check_close("sum", got, expected, 1e-5, 0.0);

  // The kernel reduces in place, so timed runs operate on already-reduced
  // data. The memory access pattern, and therefore the time, is the same.
  const float ms = lab::time_ms(
      [&] { reduction<<<numBlocks, BLOCK_SIZE>>>(d_input, d_partialSums, N); });
  lab::report("naive reduction", ms, N, static_cast<double>(N) * sizeof(float));

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_partialSums));
  return lab::finish(pass);
}
