/*
 * Histogram 1: Privatization in Shared Memory
 *
 * Intention:
 * Give every block its own private copy of the histogram in shared memory.
 * Shared-memory atomics are much faster than global ones, and contention is
 * limited to the threads of one block instead of the whole GPU. At the end,
 * each block adds its private histogram into the global one: 256 global
 * atomics per block instead of one per element.
 *
 * High-Level Algorithm:
 * - Zero a 256-bin shared histogram; __syncthreads().
 * - Grid-stride loop: atomicAdd on the shared bin for every element.
 * - __syncthreads(); each thread merges one or more shared bins into the
 *   global histogram (skipping empty bins).
 *
 * This "privatization" pattern applies to any reduction into a small output:
 * do most of the work on a private copy, merge once.
 */
#include "../histogram_harness.cuh"

constexpr int THREADS = 256;

__global__ void histogram_shared(const unsigned char *in, int n,
                                 unsigned int *hist) {
  __shared__ unsigned int local[NUM_BINS];
  for (int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
    local[b] = 0;
  }
  __syncthreads();

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    atomicAdd(&local[in[i]], 1u);
  }
  __syncthreads();

  for (int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
    if (local[b] != 0) {
      atomicAdd(&hist[b], local[b]);
    }
  }
}

void launch(const unsigned char *d_in, int n, unsigned int *d_hist, void *,
            size_t) {
  histogram_shared<<<histogram_blocks(n, THREADS), THREADS>>>(d_in, n, d_hist);
}

int main(int argc, char **argv) {
  return run_histogram("1. Shared-memory privatization", argc, argv, launch);
}
