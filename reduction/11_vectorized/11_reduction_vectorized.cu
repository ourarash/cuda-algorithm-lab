/*
 * Reduction 11: Vectorized Loads (float4)
 *
 * Intention:
 * Step 10 with 128-bit loads. Each thread reads four floats per load
 * instruction instead of one, so the same bytes need a quarter of the load
 * instructions and loop iterations, which helps the memory system stay busy
 * enough to reach peak bandwidth.
 *
 * High-Level Algorithm:
 * - View the input as float4 (cudaMalloc returns 256-byte-aligned memory, so
 *   the reinterpretation is safe) and run the grid-stride loop over n / 4
 *   vectors.
 * - The last n % 4 elements do not fill a float4; the first few threads of
 *   the grid add them with scalar loads.
 * - Block reduction with shuffles and the "last block finishes" single pass
 *   from step 10.
 */
#include "../reduction_harness.cuh"

constexpr int BLOCK_SIZE = 256;
constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

__device__ unsigned int blocks_done = 0;

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    v += __shfl_down_sync(FULL_MASK, v, offset);
  }
  return v;
}

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

__global__ void reduce_vectorized(const float *in, float *partials, float *out,
                                  int n) {
  __shared__ bool is_last_block;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  float sum = 0.0f;
  const float4 *in4 = reinterpret_cast<const float4 *>(in);
  const int n4 = n / 4;
  for (int i = tid; i < n4; i += stride) {
    const float4 v = in4[i];
    sum += (v.x + v.y) + (v.z + v.w);
  }
  for (int i = 4 * n4 + tid; i < n; i += stride) {  // The n % 4 leftovers
    sum += in[i];
  }
  sum = block_reduce_sum(sum);

  if (threadIdx.x == 0) {
    partials[blockIdx.x] = sum;
    __threadfence();
    is_last_block = atomicAdd(&blocks_done, 1) == gridDim.x - 1;
  }
  __syncthreads();

  if (is_last_block) {
    float total = 0.0f;
    for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
      total += __ldcg(&partials[i]);
    }
    total = block_reduce_sum(total);
    if (threadIdx.x == 0) {
      *out = total;
      blocks_done = 0;
    }
  }
}

void launch(float *d_in, int n, float *d_out, float *d_scratch) {
  const int blocks = grid_stride_blocks(lab::ceil_div(n, 4), BLOCK_SIZE);
  reduce_vectorized<<<blocks, BLOCK_SIZE>>>(d_in, d_scratch, d_out, n);
}

int main(int argc, char **argv) {
  return run_reduction("11. Vectorized (float4)", argc, argv, launch);
}
