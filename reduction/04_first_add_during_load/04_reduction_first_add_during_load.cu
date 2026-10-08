/*
 * Reduction 4: First Add During Load
 *
 * Intention:
 * Step 4 of Harris's sequence. In step 3, half the threads go idle after the
 * load. Here every thread loads two elements and adds them before writing to
 * shared memory, so each block reduces 2 * blockDim elements with the same
 * tree and the grid needs half as many blocks.
 *
 * High-Level Algorithm:
 * - Thread tid of block b loads elements b * 2B + tid and b * 2B + tid + B
 *   (B = blockDim), adds them, and stores the sum in sdata[tid]. Both loads
 *   are coalesced.
 * - Sequential-addressing tree as in step 3.
 *
 * What is left:
 * With bandwidth used better, instruction overhead starts to show: the loop,
 * the __syncthreads() at every level, and the address arithmetic. The next
 * steps remove those.
 */
#include "../reduction_harness.cuh"

constexpr int BLOCK_SIZE = 256;

__global__ void reduce_first_add_during_load(const float *in, float *out, int n) {
  __shared__ float sdata[BLOCK_SIZE];
  const unsigned int tid = threadIdx.x;
  const unsigned int i = blockIdx.x * (2 * BLOCK_SIZE) + tid;
  float sum = i < n ? in[i] : 0.0f;
  if (i + BLOCK_SIZE < n) {
    sum += in[i + BLOCK_SIZE];
  }
  sdata[tid] = sum;
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
  reduce_in_passes(d_in, n, d_out, d_scratch, 2 * BLOCK_SIZE,
                   [](float *src, float *dst, int count, int blocks) {
                     reduce_first_add_during_load<<<blocks, BLOCK_SIZE>>>(src, dst, count);
                   });
}

int main(int argc, char **argv) {
  return run_reduction("4. First add during load", argc, argv, launch);
}
