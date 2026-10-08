/*
 * Bitonic Sort
 *
 * Intention:
 * Bitonic sort is a sorting network: a fixed sequence of compare-and-swap
 * steps that does not depend on the data. That makes it a natural fit for
 * GPUs (no divergence, perfectly regular memory access) even though it does
 * O(n log^2 n) work, more than an O(n log n) comparison sort.
 *
 * The network:
 * - For size = 2, 4, 8, ..., P (P = array length, a power of two):
 *     for stride = size / 2, size / 4, ..., 1:
 *       every element i whose `stride` bit is 0 is compared with i + stride
 *       and the pair is put in ascending order if (i & size) == 0, else in
 *       descending order.
 * - After stage `size`, every run of `size` elements is sorted (alternately
 *   ascending and descending, which is exactly the "bitonic" input the next
 *   stage needs); after the last stage the whole array is ascending.
 *
 * GPU implementation (the structure of the CUDA samples' sortingNetworks):
 * - Steps with stride < 1024 only touch elements within an aligned
 *   1024-element chunk, so they run in shared memory: bitonic_sort_shared
 *   does every stage up to size 1024, and bitonic_merge_shared does the small
 *   strides of each later stage.
 * - Steps with stride >= 1024 cross chunks and run as one global-memory
 *   kernel per step (bitonic_merge_global), one thread per pair.
 * - Arbitrary n: the array is padded to a power of two with the largest key,
 *   which sorts to the end and is not copied back.
 */
#include <climits>

#include "../sort_harness.cuh"

constexpr int THREADS = 512;
constexpr int CHUNK = 2 * THREADS;  // Elements sorted per block in shared memory

__device__ __forceinline__ void compare_exchange(unsigned int &a,
                                                 unsigned int &b,
                                                 bool ascending) {
  if ((a > b) == ascending) {
    const unsigned int t = a;
    a = b;
    b = t;
  }
}

// Lower index of pair `p` at the given stride: insert a 0 at bit `stride`.
__device__ __forceinline__ int pair_low(int p, int stride) {
  return 2 * stride * (p / stride) + (p % stride);
}

// All stages with size <= CHUNK, inside each CHUNK-element chunk.
__global__ void bitonic_sort_shared(unsigned int *data) {
  __shared__ unsigned int s[CHUNK];
  const int base = blockIdx.x * CHUNK;
  s[threadIdx.x] = data[base + threadIdx.x];
  s[threadIdx.x + THREADS] = data[base + threadIdx.x + THREADS];
  __syncthreads();

  for (int size = 2; size <= CHUNK; size *= 2) {
    for (int stride = size / 2; stride > 0; stride /= 2) {
      const int low = pair_low(threadIdx.x, stride);
      const bool ascending = ((base + low) & size) == 0;
      compare_exchange(s[low], s[low + stride], ascending);
      __syncthreads();
    }
  }

  data[base + threadIdx.x] = s[threadIdx.x];
  data[base + threadIdx.x + THREADS] = s[threadIdx.x + THREADS];
}

// One step of stage `size` with stride >= CHUNK; one thread per pair.
__global__ void bitonic_merge_global(unsigned int *data, int size, int stride) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  const int low = pair_low(p, stride);
  const bool ascending = (low & size) == 0;
  unsigned int a = data[low];
  unsigned int b = data[low + stride];
  compare_exchange(a, b, ascending);
  data[low] = a;
  data[low + stride] = b;
}

// The steps of stage `size` with stride < CHUNK, inside each chunk.
__global__ void bitonic_merge_shared(unsigned int *data, int size) {
  __shared__ unsigned int s[CHUNK];
  const int base = blockIdx.x * CHUNK;
  s[threadIdx.x] = data[base + threadIdx.x];
  s[threadIdx.x + THREADS] = data[base + threadIdx.x + THREADS];
  __syncthreads();

  for (int stride = CHUNK / 2; stride > 0; stride /= 2) {
    const int low = pair_low(threadIdx.x, stride);
    const bool ascending = ((base + low) & size) == 0;
    compare_exchange(s[low], s[low + stride], ascending);
    __syncthreads();
  }

  data[base + threadIdx.x] = s[threadIdx.x];
  data[base + threadIdx.x + THREADS] = s[threadIdx.x + THREADS];
}

void launch(const unsigned int *d_in, unsigned int *d_out, int n,
            void *d_scratch, size_t scratch_bytes) {
  int padded = CHUNK;
  while (padded < n) {
    padded *= 2;
  }
  if (padded * sizeof(unsigned int) > scratch_bytes) {
    std::fprintf(stderr, "bitonic sort needs %zu bytes of scratch\n",
                 padded * sizeof(unsigned int));
    std::exit(EXIT_FAILURE);
  }
  unsigned int *buf = static_cast<unsigned int *>(d_scratch);
  CUDA_CHECK(cudaMemcpyAsync(buf, d_in, n * sizeof(unsigned int),
                             cudaMemcpyDeviceToDevice));
  // Pad with 0xFFFFFFFF (UINT_MAX), which sorts to the end.
  CUDA_CHECK(cudaMemsetAsync(buf + n, 0xFF, (padded - n) * sizeof(unsigned int)));

  const int chunks = padded / CHUNK;
  bitonic_sort_shared<<<chunks, THREADS>>>(buf);
  for (int size = 2 * CHUNK; size <= padded; size *= 2) {
    for (int stride = size / 2; stride >= CHUNK; stride /= 2) {
      bitonic_merge_global<<<padded / 2 / 256, 256>>>(buf, size, stride);
    }
    bitonic_merge_shared<<<chunks, THREADS>>>(buf, size);
  }
  CUDA_CHECK(cudaMemcpyAsync(d_out, buf, n * sizeof(unsigned int),
                             cudaMemcpyDeviceToDevice));
}

int main(int argc, char **argv) {
  return run_sort("Bitonic sort", argc, argv, launch);
}
