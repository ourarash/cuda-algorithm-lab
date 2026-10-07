/*
 * Shared Memory Parallel Reduction (Optimized)
 *
 * High-Level Algorithm:
 * This kernel optimizes the naive reduction by employing fast on-chip shared memory,
 * sequential thread mapping, and a redesigned reduction tree to eliminate warp divergence.
 *
 * Phase 1 (Global-to-Register Accumulation):
 * Each thread reads multiple elements from global memory, accumulating them
 * into a local register sum. Consecutive threads read consecutive elements,
 * so the reads are perfectly coalesced, and the extra work per thread
 * amortizes the cost of the tree that follows. The thread's partial sum is
 * then written into shared memory once.
 *
 * Phase 2 (Tree Reduction in Shared Memory):
 * How this tree drastically differs from the Naive approach:
 * - Shrinking Stride vs. Growing Stride: The naive tree started with stride=1 and grew (1, 2, 4...).
 *   This optimized tree starts at half the block size and shrinks by half (e.g., 128, 64, 32...).
 * - Packed Active Threads vs. Scattered: In the naive tree, active threads were scattered
 *   (`tid % stride == 0`), destroying warp utilization. Here, the check is `tid < stride`.
 *   This tightly packs all active threads contiguous to each other on the left side of the block.
 * - Resulting Hardware Efficiency: Because active threads are grouped together (0 to stride - 1),
 *   every warp is either fully active or fully idle, which eliminates warp divergence for all
 *   steps until the number of active threads drops below 32.
 *
 * Why not a *growing* stride with packed threads in Shared Memory?
 * - If we packed threads but used a growing stride (e.g., `sharedData[2 * stride * tid] += ...`),
 *   adjacent threads (T0, T1, T2) would access memory at leaps of 2, 4, 8, etc. Since shared
 *   memory is divided into 32 banks, a stride of 2 causes 2-way bank conflicts (hardware serializes
 *   the memory reads). A shrinking stride (`sharedData[tid] += ...`) ensures adjacent threads access
 *   adjacent indices (stride of 1), eliminating both warp divergence AND bank conflicts.
 */
#include <cstdio>
#include <vector>

#include "lab.cuh"

#define BLOCK_SIZE 256  // Number of threads per block (must be a power of two)
// Number of BLOCK_SIZE-wide segments each thread block reduces. Each thread
// adds up this many input elements in Phase 1.
#define INPUT_SEGMENTS_PER_THREAD_BLOCK 4

// -------------------------------------------------------------------------
// Shared Memory Parallel Reduction Kernel
// -------------------------------------------------------------------------
// This reduction kernel improves on the naive approach by utilizing fast
// on-chip shared memory, contiguous memory accesses, and reducing warp
// divergence. Instead of modifying global memory, the threads first load data
// and accumulate an initial sum, then perform a tree reduction in shared
// memory.
__global__ void reduction(const float* input, float* partialSums,
                          unsigned int N) {
  // Local thread ID within the block
  unsigned int tid = threadIdx.x;

  // Global starting index for this thread. Each thread block handles
  // INPUT_SEGMENTS_PER_THREAD_BLOCK consecutive segments of blockDim.x
  // elements.
  unsigned int i =
      blockIdx.x * blockDim.x * INPUT_SEGMENTS_PER_THREAD_BLOCK + tid;

  // Shared memory for this block: one partial sum per thread.
  __shared__ float sharedData[BLOCK_SIZE];

  // Phase 1: Global-to-Register Accumulation
  // Each thread sums one element from each of the block's segments in a
  // register. Within one iteration, consecutive threads read consecutive
  // addresses, so every warp's load is coalesced.
  float sum = 0.0f;
#pragma unroll
  for (int j = 0; j < INPUT_SEGMENTS_PER_THREAD_BLOCK; ++j) {
    if (i + j * blockDim.x < N) {
      sum += input[i + j * blockDim.x];
    }
  }
  sharedData[tid] = sum;

  // Ensure all threads have finished writing their initial sums to shared
  // memory.
  __syncthreads();

  // Phase 2: Tree-based Reduction in Shared Memory
  // We use a shrinking stride (stride /= 2) instead of a growing stride.
  // This avoids warp divergence because active threads share the same
  // consecutive warps (e.g., in the first iteration, threads 0 to 127 are
  // active, spanning warps 0-3 fully).
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      // Add the value from the right half to the left half in shared memory.
      sharedData[tid] += sharedData[tid + stride];
    }
    // Block-wide synchronization is required at each depth of the tree.
    __syncthreads();
  }

  // At the end of the loop, sharedData[0] contains the sum for the entire
  // block. Thread 0 writes this value to the output array holding partial sums.
  if (tid == 0) {
    partialSums[blockIdx.x] = sharedData[0];
  }
}

int main(int argc, char** argv) {
  lab::Args args(argc, argv);
  const unsigned int N = static_cast<unsigned int>(
      args.get_int("n", args.quick() ? 100003 : (1 << 24) + 123));
  const unsigned int numBlocks =
      lab::ceil_div(N, static_cast<unsigned int>(INPUT_SEGMENTS_PER_THREAD_BLOCK * BLOCK_SIZE));

  lab::print_device();
  printf("Shared-memory reduction of %u floats\n", N);

  const std::vector<float> h_input = lab::random_uniform<float>(N, 0.f, 1.f, 7);
  float *d_input, *d_partialSums;
  CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_partialSums, numBlocks * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float),
                        cudaMemcpyHostToDevice));

  reduction<<<numBlocks, BLOCK_SIZE>>>(d_input, d_partialSums, N);
  CUDA_CHECK_LAUNCH();

  // Add the block partial sums on the host in double precision.
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
  const std::vector<double> got = {gpu_sum}, expected = {cpu_sum};
  const bool pass = lab::check_close("sum", got, expected, 1e-5, 0.0);

  const float ms = lab::time_ms(
      [&] { reduction<<<numBlocks, BLOCK_SIZE>>>(d_input, d_partialSums, N); });
  lab::report("shared-memory reduction", ms, N,
              static_cast<double>(N) * sizeof(float));

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_partialSums));
  return lab::finish(pass);
}
