/*
 * Reduction 9: Cooperative Groups
 *
 * Intention:
 * The same algorithm as step 8, written with the Cooperative Groups API
 * instead of raw shuffles. Groups make the set of threads that cooperate
 * explicit (a block, a 32-thread tile), and cg::reduce picks the best
 * hardware path for the group (on compute capability 8.0+ that includes the
 * warp-wide reduction instructions for integer types; floats use shuffles).
 *
 * High-Level Algorithm:
 * - Grid-stride loop into a register, as before.
 * - cg::reduce over the 32-thread tile gives each warp's sum.
 * - Tile leaders write to shared memory; after block.sync(), the first tile
 *   reduces the per-warp sums with cg::reduce again.
 * - Two launches, as in steps 7 and 8.
 */
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "../reduction_harness.cuh"

namespace cg = cooperative_groups;

constexpr int BLOCK_SIZE = 256;

__global__ void reduce_cooperative_groups(const float *in, float *out, int n) {
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
  __shared__ float warp_sums[32];

  float sum = 0.0f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    sum += in[i];
  }

  sum = cg::reduce(warp, sum, cg::plus<float>());
  if (warp.thread_rank() == 0) {
    warp_sums[warp.meta_group_rank()] = sum;
  }
  block.sync();

  if (warp.meta_group_rank() == 0) {
    float v = warp.thread_rank() < warp.meta_group_size()
                  ? warp_sums[warp.thread_rank()]
                  : 0.0f;
    v = cg::reduce(warp, v, cg::plus<float>());
    if (warp.thread_rank() == 0) {
      out[blockIdx.x] = v;
    }
  }
}

void launch(float *d_in, int n, float *d_out, float *d_scratch) {
  const int blocks = grid_stride_blocks(n, BLOCK_SIZE);
  reduce_cooperative_groups<<<blocks, BLOCK_SIZE>>>(d_in, d_scratch, n);
  reduce_cooperative_groups<<<1, BLOCK_SIZE>>>(d_scratch, d_out, blocks);
}

int main(int argc, char **argv) {
  return run_reduction("9. Cooperative groups", argc, argv, launch);
}
