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
 * Each pass turns n values into one partial sum per block; the launcher
 * repeats passes until one value remains, so the whole reduction runs on the
 * GPU. The input sizes are deliberately not multiples of the 2 * BLOCK_SIZE
 * elements each block handles, so the bounds checks are exercised.
 *
 * The host-side driver (inputs, validation, timing) lives in
 * ../reduction_harness.cuh and is shared by every step in this folder.
 */
#include "../reduction_harness.cuh"

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

void launch_reduction(float *d_in, int n, float *d_out, float *d_scratch) {
  reduce_in_passes(d_in, n, d_out, d_scratch, 2 * BLOCK_SIZE,
                   [](float *src, float *dst, int count, int blocks) {
                     reduction<<<blocks, BLOCK_SIZE>>>(src, dst, count);
                   });
}

int main(int argc, char **argv) {
  return run_reduction("Naive (global memory)", argc, argv, launch_reduction);
}
