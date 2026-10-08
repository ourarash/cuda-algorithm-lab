/*
 * Histogram 2: Vector Loads and Aggregation
 *
 * Intention:
 * Two refinements on top of shared-memory privatization:
 * - Coarsening with 16-byte loads: each thread reads 16 consecutive bytes
 *   with one uint4 load instead of 16 one-byte loads.
 * - Aggregation: when consecutive values in a thread's 16 bytes are equal (a
 *   run, common in images), the thread counts the run in a register and
 *   issues one atomicAdd for the whole run instead of one per element. This
 *   helps exactly where contention hurts most: data where many neighbouring
 *   values fall into the same bin.
 *
 * High-Level Algorithm:
 * - Private shared histogram as in step 1.
 * - Grid-stride loop over 16-byte chunks. For each chunk, walk its bytes,
 *   extending the current run while the value repeats and flushing it with a
 *   single shared atomicAdd when the value changes. The last bytes that do
 *   not fill a chunk are handled one by one.
 * - Merge into the global histogram as in step 1.
 *
 * On uniform data runs are rare, so aggregation costs a compare per byte and
 * saves little; on the skewed input it removes most of the atomics.
 */
#include "../histogram_harness.cuh"

constexpr int THREADS = 256;

__device__ __forceinline__ void count_bytes(unsigned int word, int &run_value,
                                            unsigned int &run_length,
                                            unsigned int *local) {
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int value = (word >> (8 * k)) & 0xFF;
    if (value == run_value) {
      ++run_length;
    } else {
      if (run_length > 0) {
        atomicAdd(&local[run_value], run_length);
      }
      run_value = value;
      run_length = 1;
    }
  }
}

__global__ void histogram_aggregated(const unsigned char *in, int n,
                                     unsigned int *hist) {
  __shared__ unsigned int local[NUM_BINS];
  for (int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
    local[b] = 0;
  }
  __syncthreads();

  // cudaMalloc returns 256-byte-aligned memory, so in[] can be read as uint4.
  const uint4 *in16 = reinterpret_cast<const uint4 *>(in);
  const int chunks = n / 16;
  int run_value = -1;
  unsigned int run_length = 0;
  for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < chunks;
       c += blockDim.x * gridDim.x) {
    const uint4 v = in16[c];
    count_bytes(v.x, run_value, run_length, local);
    count_bytes(v.y, run_value, run_length, local);
    count_bytes(v.z, run_value, run_length, local);
    count_bytes(v.w, run_value, run_length, local);
  }
  // The n % 16 bytes past the last full chunk.
  for (int i = chunks * 16 + blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    atomicAdd(&local[in[i]], 1u);
  }
  if (run_length > 0) {
    atomicAdd(&local[run_value], run_length);
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
  const int blocks = histogram_blocks(lab::ceil_div(n, 16), THREADS);
  histogram_aggregated<<<blocks, THREADS>>>(d_in, n, d_hist);
}

int main(int argc, char **argv) {
  return run_histogram("2. Vector loads + aggregation", argc, argv, launch);
}
