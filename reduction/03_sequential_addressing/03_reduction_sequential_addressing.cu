/*
 * Reduction 3: Sequential Addressing
 *
 * Intention:
 * Step 3 of Harris's sequence. Fix the bank conflicts of step 2 by reversing
 * the tree: start with a large stride and halve it.
 *
 * High-Level Algorithm:
 * - At stride s = blockDim / 2, blockDim / 4, ..., 1: threads tid < s add
 *   sdata[tid + s] into sdata[tid].
 * - Active threads are consecutive (no divergence until fewer than 32 are
 *   left) and a warp reads 32 consecutive words (no bank conflicts).
 *
 * What is wasteful now:
 * Half of the threads are idle from the very first step: they only load one
 * element and then watch. The next step gives them real work during the load.
 *
 * shared_reduction_visualization.html in this folder animates this tree.
 */
#include "../reduction_harness.cuh"

constexpr int BLOCK_SIZE = 256;

__global__ void reduce_sequential_addressing(const float *in, float *out, int n) {
  __shared__ float sdata[BLOCK_SIZE];
  const unsigned int tid = threadIdx.x;
  const unsigned int i = blockIdx.x * BLOCK_SIZE + tid;
  sdata[tid] = i < n ? in[i] : 0.0f;
  __syncthreads();

  for (unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    out[blockIdx.x] = sdata[0];
  }
}

void launch(float *d_in, int n, float *d_out, float *d_scratch) {
  reduce_in_passes(d_in, n, d_out, d_scratch, BLOCK_SIZE,
                   [](float *src, float *dst, int count, int blocks) {
                     reduce_sequential_addressing<<<blocks, BLOCK_SIZE>>>(src, dst, count);
                   });
}

int main(int argc, char **argv) {
  return run_reduction("3. Sequential addressing", argc, argv, launch);
}
