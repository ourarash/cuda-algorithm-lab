/*
 * Reduction 10: Single Pass ("Last Block Finishes")
 *
 * Intention:
 * Steps 7-9 need two kernel launches: one for the per-block sums and one to
 * add those up. A launch costs a few microseconds, which is a noticeable
 * share of the time for mid-sized inputs. This step finishes in one launch:
 * whichever block completes last adds up all the block sums.
 *
 * High-Level Algorithm:
 * - Grid-stride loop and shuffle-based block reduction as in step 8; thread 0
 *   of each block writes the block's sum to partials[blockIdx.x].
 * - Thread 0 then executes __threadfence() and takes a ticket with
 *   atomicAdd(&blocks_done, 1). The fence guarantees that the partial sum
 *   is visible to every other block before the ticket is.
 * - The block that draws ticket gridDim.x - 1 knows every other partial sum
 *   is already in memory. It reduces them, writes the result, and resets the
 *   counter for the next launch.
 *
 * Why not atomicAdd every block's sum into the result? That is also a single
 * pass, but floating-point addition is not associative and the order of
 * atomics changes from run to run, so the result would not be reproducible
 * bit for bit. Here the final additions happen in a fixed order.
 */
#include "../reduction_harness.cuh"

constexpr int BLOCK_SIZE = 256;
constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

// Counts finished blocks; the last block resets it.
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

__global__ void reduce_single_pass(const float *in, float *partials,
                                   float *out, int n) {
  __shared__ bool is_last_block;

  float sum = 0.0f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    sum += in[i];
  }
  sum = block_reduce_sum(sum);

  if (threadIdx.x == 0) {
    partials[blockIdx.x] = sum;
    __threadfence();  // Publish the partial sum before taking a ticket.
    const unsigned int ticket = atomicAdd(&blocks_done, 1);
    is_last_block = ticket == gridDim.x - 1;
  }
  __syncthreads();

  if (is_last_block) {
    float total = 0.0f;
    for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
      // __ldcg reads through L2, never a stale L1 copy.
      total += __ldcg(&partials[i]);
    }
    total = block_reduce_sum(total);
    if (threadIdx.x == 0) {
      *out = total;
      blocks_done = 0;  // Ready for the next launch.
    }
  }
}

void launch(float *d_in, int n, float *d_out, float *d_scratch) {
  const int blocks = grid_stride_blocks(n, BLOCK_SIZE);
  reduce_single_pass<<<blocks, BLOCK_SIZE>>>(d_in, d_scratch, d_out, n);
}

int main(int argc, char **argv) {
  return run_reduction("10. Single pass", argc, argv, launch);
}
