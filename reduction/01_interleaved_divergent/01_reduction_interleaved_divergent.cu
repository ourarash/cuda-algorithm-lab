/*
 * Reduction 1: Interleaved Addressing with Divergent Branching
 *
 * Intention:
 * The first shared-memory reduction, following step 1 of Mark Harris's
 * "Optimizing Parallel Reduction in CUDA". Each block copies its slice of the
 * input into shared memory and reduces it there as a binary tree, so the
 * intermediate sums never touch global memory (unlike 00_naive).
 *
 * High-Level Algorithm:
 * - Thread tid loads one element into sdata[tid].
 * - At stride s = 1, 2, 4, ...: threads whose tid is a multiple of 2s add
 *   sdata[tid + s] into sdata[tid].
 * - Thread 0 writes the block's sum; the launcher repeats passes over the
 *   partial sums until one value remains.
 *
 * What is slow:
 * `tid % (2 * s) == 0` selects threads that are spread out, so at stride 1
 * only every second thread of each warp works, at stride 2 every fourth, and
 * so on. A warp executes both sides of the branch, so most lanes sit idle
 * (warp divergence), and the % operator itself is slow.
 */
#include "../reduction_harness.cuh"

constexpr int BLOCK_SIZE = 256;

__global__ void reduce_interleaved_divergent(const float *in, float *out, int n) {
  __shared__ float sdata[BLOCK_SIZE];
  const unsigned int tid = threadIdx.x;
  const unsigned int i = blockIdx.x * BLOCK_SIZE + tid;
  sdata[tid] = i < n ? in[i] : 0.0f;
  __syncthreads();

  for (unsigned int s = 1; s < BLOCK_SIZE; s *= 2) {
    if (tid % (2 * s) == 0) {
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
                     reduce_interleaved_divergent<<<blocks, BLOCK_SIZE>>>(src, dst, count);
                   });
}

int main(int argc, char **argv) {
  return run_reduction("1. Interleaved, divergent", argc, argv, launch);
}
