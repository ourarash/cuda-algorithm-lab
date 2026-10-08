/*
 * Reduction 6: Complete Unrolling
 *
 * Intention:
 * Step 6 of Harris's sequence. If the block size is known at compile time,
 * the whole tree can be unrolled: no loop counter, no loop branch, and every
 * `if (BLOCK_SIZE >= ...)` test on a template parameter is resolved by the
 * compiler, leaving only the work that this block size needs.
 *
 * High-Level Algorithm:
 * - The block size is a template parameter. The launcher instantiates the
 *   kernel for the block size it uses (production code typically dispatches
 *   over 64, 128, 256, 512, 1024 with a switch).
 * - Block-wide levels are written out one by one, each guarded by a
 *   compile-time test, then warp 0 finishes with the __syncwarp() sequence
 *   from step 5.
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
__global__ void reduce_complete_unroll(const float *in, float *out, int n) {
  static_assert(BS >= 64 && BS <= 1024 && (BS & (BS - 1)) == 0,
                "power-of-two block size from 64 to 1024");
  __shared__ float sdata[BS];
  const unsigned int tid = threadIdx.x;
  const unsigned int i = blockIdx.x * (2 * BS) + tid;
  float sum = i < n ? in[i] : 0.0f;
  if (i + BS < n) {
    sum += in[i + BS];
  }
  sdata[tid] = sum;
  __syncthreads();

  if (BS >= 1024) {
    if (tid < 512) sdata[tid] += sdata[tid + 512];
    __syncthreads();
  }
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
  reduce_in_passes(d_in, n, d_out, d_scratch, 2 * BLOCK_SIZE,
                   [](float *src, float *dst, int count, int blocks) {
                     reduce_complete_unroll<BLOCK_SIZE><<<blocks, BLOCK_SIZE>>>(src, dst, count);
                   });
}

int main(int argc, char **argv) {
  return run_reduction("6. Complete unroll", argc, argv, launch);
}
