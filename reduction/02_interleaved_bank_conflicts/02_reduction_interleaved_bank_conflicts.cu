/*
 * Reduction 2: Interleaved Addressing without Divergence (Bank Conflicts)
 *
 * Intention:
 * Step 2 of Harris's sequence. Remove the divergent branch of step 1 by
 * giving the work to consecutive threads instead of scattered ones.
 *
 * High-Level Algorithm:
 * - Same tree as step 1, but at stride s thread tid works on
 *   index = 2 * s * tid: sdata[index] += sdata[index + s].
 * - The active threads are now 0, 1, 2, ... so whole warps are either busy
 *   or idle, and there is no % operator.
 *
 * What is slow now:
 * Consecutive threads access shared memory 2s words apart. Shared memory has
 * 32 banks of 4-byte words, so at stride 1 two threads of a warp hit each
 * bank, at stride 2 four, and so on: 2-way, 4-way, ... bank conflicts, which
 * the hardware serializes.
 */
#include "../reduction_harness.cuh"

constexpr int BLOCK_SIZE = 256;

__global__ void reduce_interleaved_bank_conflicts(const float *in, float *out, int n) {
  __shared__ float sdata[BLOCK_SIZE];
  const unsigned int tid = threadIdx.x;
  const unsigned int i = blockIdx.x * BLOCK_SIZE + tid;
  sdata[tid] = i < n ? in[i] : 0.0f;
  __syncthreads();

  for (unsigned int s = 1; s < BLOCK_SIZE; s *= 2) {
    const unsigned int index = 2 * s * tid;
    if (index < BLOCK_SIZE) {
      sdata[index] += sdata[index + s];
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
                     reduce_interleaved_bank_conflicts<<<blocks, BLOCK_SIZE>>>(src, dst, count);
                   });
}

int main(int argc, char **argv) {
  return run_reduction("2. Interleaved, bank conflicts", argc, argv, launch);
}
