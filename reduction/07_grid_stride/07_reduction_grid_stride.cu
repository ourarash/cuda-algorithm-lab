/*
 * Reduction 7: Multiple Elements per Thread (Grid-Stride Loop)
 *
 * Intention:
 * Step 7 of Harris's sequence. Instead of one block per 512 elements (tens of
 * thousands of blocks, each paying the full tree cost), launch only enough
 * blocks to fill the GPU and let every thread add many elements in a register
 * first. The tree then runs once per block instead of once per 512 elements.
 *
 * High-Level Algorithm:
 * - Grid size = 8 blocks per SM (or fewer for small inputs).
 * - Each thread walks the array with a grid-stride loop, two coalesced loads
 *   per iteration, accumulating in a register.
 * - The completely unrolled tree from step 6 reduces the block's partial sums.
 * - Two launches in total: the main pass leaves one value per block, and a
 *   single block reduces those.
 *
 * This is the last step of Harris's original sequence. The steps that follow
 * use hardware features added after it was written (2007).
 */
#include "../reduction_harness.cuh"

constexpr int BLOCK_SIZE = 256;

__device__ __forceinline__ void warp_reduce_shared(volatile float *sdata,
                                                   unsigned int tid) {
  float v = sdata[tid];
  v += sdata[tid + 32];
  __syncwarp();
  sdata[tid] = v;
  __syncwarp();
  v += sdata[tid + 16];
  __syncwarp();
  sdata[tid] = v;
  __syncwarp();
  v += sdata[tid + 8];
  __syncwarp();
  sdata[tid] = v;
  __syncwarp();
  v += sdata[tid + 4];
  __syncwarp();
  sdata[tid] = v;
  __syncwarp();
  v += sdata[tid + 2];
  __syncwarp();
  sdata[tid] = v;
  __syncwarp();
  v += sdata[tid + 1];
  __syncwarp();
  sdata[tid] = v;
}

template <unsigned int BS>
__global__ void reduce_grid_stride(const float *in, float *out, int n) {
  __shared__ float sdata[BS];
  const unsigned int tid = threadIdx.x;
  const unsigned int grid_size = 2 * BS * gridDim.x;

  float sum = 0.0f;
  for (unsigned int i = blockIdx.x * (2 * BS) + tid; i < n; i += grid_size) {
    sum += in[i];
    if (i + BS < n) {
      sum += in[i + BS];
    }
  }
  sdata[tid] = sum;
  __syncthreads();

  if (BS >= 512) {
    if (tid < 256) sdata[tid] += sdata[tid + 256];
    __syncthreads();
  }
  if (BS >= 256) {
    if (tid < 128) sdata[tid] += sdata[tid + 128];
    __syncthreads();
  }
  if (BS >= 128) {
    if (tid < 64) sdata[tid] += sdata[tid + 64];
    __syncthreads();
  }
  if (tid < 32) {
    warp_reduce_shared(sdata, tid);
  }
  if (tid == 0) {
    out[blockIdx.x] = sdata[0];
  }
}

void launch(float *d_in, int n, float *d_out, float *d_scratch) {
  const int blocks = grid_stride_blocks(n, 2 * BLOCK_SIZE);
  reduce_grid_stride<BLOCK_SIZE><<<blocks, BLOCK_SIZE>>>(d_in, d_scratch, n);
  reduce_grid_stride<BLOCK_SIZE><<<1, BLOCK_SIZE>>>(d_scratch, d_out, blocks);
}

int main(int argc, char **argv) {
  return run_reduction("7. Grid-stride loop", argc, argv, launch);
}
