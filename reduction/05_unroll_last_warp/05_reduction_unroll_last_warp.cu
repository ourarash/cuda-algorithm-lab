/*
 * Reduction 5: Unroll the Last Warp
 *
 * Intention:
 * Step 5 of Harris's sequence. Once only 32 or fewer threads are active,
 * they all belong to one warp, so the block-wide __syncthreads() and the
 * `if (tid < s)` test are wasted work for the other warps. The last six
 * levels of the tree are written out by hand and run by warp 0 alone.
 *
 * Modern correction:
 * Harris's original code relied on the 32 threads of a warp executing in
 * lockstep and used a `volatile` pointer with no synchronization at all.
 * Since Volta (2017), threads of a warp are scheduled independently and may
 * diverge, so that code is no longer correct. Each step below separates the
 * shared-memory reads from the writes with __syncwarp(), which is cheap: it
 * only synchronizes the lanes of one warp.
 *
 * Steps 08 and 09 replace this shared-memory dance entirely with warp
 * shuffles, which exchange registers directly.
 */
#include "../reduction_harness.cuh"

constexpr int BLOCK_SIZE = 256;
static_assert(BLOCK_SIZE >= 64, "the warp phase reads sdata[tid + 32]");

// Final 64 -> 1 values, run by the 32 threads of warp 0.
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

__global__ void reduce_unroll_last_warp(const float *in, float *out, int n) {
  __shared__ float sdata[BLOCK_SIZE];
  const unsigned int tid = threadIdx.x;
  const unsigned int i = blockIdx.x * (2 * BLOCK_SIZE) + tid;
  float sum = i < n ? in[i] : 0.0f;
  if (i + BLOCK_SIZE < n) {
    sum += in[i + BLOCK_SIZE];
  }
  sdata[tid] = sum;
  __syncthreads();

  // Block-wide levels, down to 64 values.
  for (unsigned int s = BLOCK_SIZE / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // Warp-wide levels, 64 -> 1.
  if (tid < 32) {
    warp_reduce_shared(sdata, tid);
  }

  if (tid == 0) {
    out[blockIdx.x] = sdata[0];
  }
}

void launch(float *d_in, int n, float *d_out, float *d_scratch) {
  reduce_in_passes(d_in, n, d_out, d_scratch, 2 * BLOCK_SIZE,
                   [](float *src, float *dst, int count, int blocks) {
                     reduce_unroll_last_warp<<<blocks, BLOCK_SIZE>>>(src, dst, count);
                   });
}

int main(int argc, char **argv) {
  return run_reduction("5. Unroll last warp", argc, argv, launch);
}
