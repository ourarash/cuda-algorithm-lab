/*
 * Histogram 0: Global Atomics
 *
 * Intention:
 * The direct approach: every thread reads elements and increments the global
 * bin for each one with atomicAdd. Atomics make concurrent increments of the
 * same bin correct, but every one of them is an L2 round trip, and updates to
 * the same address are serialized.
 *
 * High-Level Algorithm:
 * - Grid-stride loop over the input.
 * - atomicAdd(&hist[value], 1) for every element.
 *
 * What to look for: on the skewed input, many threads hit the same few bins
 * at once, so the atomics serialize and throughput drops far below the
 * uniform input.
 */
#include "../histogram_harness.cuh"

constexpr int THREADS = 256;

__global__ void histogram_global_atomics(const unsigned char *in, int n,
                                         unsigned int *hist) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    atomicAdd(&hist[in[i]], 1u);
  }
}

void launch(const unsigned char *d_in, int n, unsigned int *d_hist, void *,
            size_t) {
  histogram_global_atomics<<<histogram_blocks(n, THREADS), THREADS>>>(d_in, n, d_hist);
}

int main(int argc, char **argv) {
  return run_histogram("0. Global atomics", argc, argv, launch);
}
