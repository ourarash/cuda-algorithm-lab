/*
 * Reduction 8: Warp Shuffles
 *
 * Intention:
 * Since Kepler, threads of a warp can read each other's registers directly
 * with shuffle instructions. A warp can sum 32 values in five steps without
 * touching shared memory or synchronizing, which replaces most of the
 * shared-memory tree of steps 1-7.
 *
 * High-Level Algorithm:
 * - Grid-stride loop as in step 7: each thread sums many elements in a
 *   register.
 * - warp_reduce_sum: __shfl_down_sync with offsets 16, 8, 4, 2, 1 adds each
 *   lane's value to the value `offset` lanes above it; after five steps lane 0
 *   holds the warp's sum.
 * - block_reduce_sum: lane 0 of each warp writes its sum to shared memory
 *   (8 values for 256 threads), one __syncthreads(), then warp 0 reduces those
 *   with shuffles again.
 * - Two launches, as in step 7.
 *
 * Shared memory traffic drops from 256 + 255 writes per block to 8, and the
 * block synchronizes once instead of at every level.
 */
#include "../reduction_harness.cuh"

constexpr int BLOCK_SIZE = 256;
constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    v += __shfl_down_sync(FULL_MASK, v, offset);
  }
  return v;  // The full sum is in lane 0.
}

// Sum over the whole block; the result is valid in thread 0.
__device__ __forceinline__ float block_reduce_sum(float v) {
  __shared__ float warp_sums[32];
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  v = warp_reduce_sum(v);
  if (lane == 0) {
    warp_sums[warp] = v;
  }
  __syncthreads();
  if (warp == 0) {
    v = lane < blockDim.x / 32 ? warp_sums[lane] : 0.0f;
    v = warp_reduce_sum(v);
  }
  return v;
}

__global__ void reduce_warp_shuffle(const float *in, float *out, int n) {
  float sum = 0.0f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    sum += in[i];
  }
  sum = block_reduce_sum(sum);
  if (threadIdx.x == 0) {
    out[blockIdx.x] = sum;
  }
}

void launch(float *d_in, int n, float *d_out, float *d_scratch) {
  const int blocks = grid_stride_blocks(n, BLOCK_SIZE);
  reduce_warp_shuffle<<<blocks, BLOCK_SIZE>>>(d_in, d_scratch, n);
  reduce_warp_shuffle<<<1, BLOCK_SIZE>>>(d_scratch, d_out, blocks);
}

int main(int argc, char **argv) {
  return run_reduction("8. Warp shuffles", argc, argv, launch);
}
