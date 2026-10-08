/*
 * Parallel Merge Sort
 *
 * Intention:
 * This file demonstrates GPU merge sort as a two-phase process: first create
 * many small sorted runs, then repeatedly merge those runs in parallel.
 *
 * High-Level Algorithm:
 * - Sort block-local chunks in shared memory with a bitonic network to
 *   create initial sorted runs.
 * - Use a co-rank based merge kernel so each thread merges a small independent
 *   slice of two sorted runs.
 * - Ping-pong between two device buffers until the entire array is sorted.
 * - Validate the result against std::sort on the CPU.
 */
#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

#include "lab.cuh"

// ===================================================================================
// Algorithm Constants
// ===================================================================================
// Each thread is responsible for merging this many elements.
#define ELEMENTS_PER_THREAD 4
// The number of threads in a CUDA block.
#define BLOCK_SIZE 256
// The total number of elements processed by a single block in one go.
#define ELEMENTS_PER_BLOCK (BLOCK_SIZE * ELEMENTS_PER_THREAD)

// ===================================================================================
// Sequential Merge (Device Function)
// ===================================================================================
// This is a standard sequential merge function that runs on the CUDA device.
// It merges two sorted arrays (A and B) into a single output array (C).
// NOTE: This is a critical correction. The output array must be passed as an
// argument.
__device__ void mergeSequential(float *A, float *B, float *C, unsigned int m,
                                unsigned int n) {
  unsigned int i = 0, j = 0, k = 0;
  while (i < m && j < n) {
    if (A[i] <= B[j]) {
      C[k++] = A[i++];
    } else {
      C[k++] = B[j++];
    }
  }
  while (i < m) {
    C[k++] = A[i++];
  }
  while (j < n) {
    C[k++] = B[j++];
  }
}

// ===================================================================================
// Co-Rank (Device Function)
// ===================================================================================
// Calculates the "co-rank", which determines how many elements from array A
// are smaller than the k-th element of the merged A and B arrays.
// This is the core of the parallel merge, allowing each thread to find its
// starting point without communicating with other threads.
__device__ unsigned int coRank(float *A, float *B, unsigned int m,
                               unsigned int n, unsigned int k) {
  int low = 0;
  int high = m;

  while (low < high) {
    int i = low + (high - low) / 2;
    int j = k - (i + 1);
    if (j < 0) {  // Went too far in A
      high = i;
      continue;
    }
    if (j >= n || A[i] <= B[j]) {
      low = i + 1;
    } else {
      high = i;
    }
  }
  return low;
}

// ===================================================================================
// Parallel Merge Kernel
// ===================================================================================
// Merges two sorted arrays (A and B) into a third array (C).
// Each thread computes a small, independent section of the merged result.
__global__ void mergeKernel(float *A, float *B, float *C, unsigned int m,
                            unsigned int n) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int k = tid * ELEMENTS_PER_THREAD;

  // Early exit if thread is out of bounds for the output array
  if (k >= m + n) {
    return;
  }

  // Use co-rank to find the start and end indices for this thread's sub-problem
  unsigned int i_start = coRank(A, B, m, n, k);
  unsigned int j_start = k - i_start;

  unsigned int k_end = min(k + ELEMENTS_PER_THREAD, m + n);
  unsigned int i_end = coRank(A, B, m, n, k_end);
  unsigned int j_end = k_end - i_end;

  // Perform a small, sequential merge on the sub-arrays identified by co-rank
  mergeSequential(A + i_start, B + j_start, C + k, i_end - i_start,
                  j_end - j_start);
}

// ===================================================================================
// Initial Sort Kernel
// ===================================================================================
// This kernel performs the first pass of the sort. Each block loads a chunk of
// ELEMENTS_PER_BLOCK values into shared memory, sorts it with a bitonic
// sorting network (see sort/02_bitonic_sort), and writes it back. This creates
// the initial sorted runs that the merge kernel works on.
//
// All 256 threads take part: each of the network's steps compares 512 pairs,
// two per thread. (An earlier version let one thread insertion-sort the whole
// chunk while the other 255 waited.) A partial last chunk is padded with
// +infinity, which sorts to the end and is not written back.

__device__ __forceinline__ void compareExchange(float &a, float &b,
                                                bool ascending) {
  if ((a > b) == ascending) {
    const float t = a;
    a = b;
    b = t;
  }
}

__global__ void initialSortKernel(float *data, unsigned int N) {
  __shared__ float shared_data[ELEMENTS_PER_BLOCK];

  unsigned int block_start_idx = blockIdx.x * ELEMENTS_PER_BLOCK;

  // Each thread loads ELEMENTS_PER_THREAD values; missing ones become +inf.
  for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
    unsigned int shared_idx = threadIdx.x + i * blockDim.x;
    unsigned int global_idx = block_start_idx + shared_idx;
    shared_data[shared_idx] = global_idx < N ? data[global_idx] : INFINITY;
  }
  __syncthreads();

  // Bitonic network over the chunk. Pair p at a given stride compares
  // elements low and low + stride, where low inserts a 0 at bit `stride`.
  for (unsigned int size = 2; size <= ELEMENTS_PER_BLOCK; size *= 2) {
    for (unsigned int stride = size / 2; stride > 0; stride /= 2) {
      for (unsigned int p = threadIdx.x; p < ELEMENTS_PER_BLOCK / 2;
           p += blockDim.x) {
        const unsigned int low = 2 * stride * (p / stride) + (p % stride);
        compareExchange(shared_data[low], shared_data[low + stride],
                        (low & size) == 0);
      }
      __syncthreads();
    }
  }

  // Write the sorted chunk from shared memory back to global memory
  for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
    unsigned int shared_idx = threadIdx.x + i * blockDim.x;
    unsigned int global_idx = block_start_idx + shared_idx;
    if (global_idx < N) {
      data[global_idx] = shared_data[shared_idx];
    }
  }
}

// ===================================================================================
// Main Host-Side Sort Function
// ===================================================================================
void parallelMergeSort(float *h_data, unsigned int N) {
  if (N == 0) return;

  // 1. Allocate memory on the device
  float *d_src, *d_dst;
  CUDA_CHECK(cudaMalloc(&d_src, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dst, N * sizeof(float)));

  // 2. Copy data from host to device source buffer
  CUDA_CHECK(
      cudaMemcpy(d_src, h_data, N * sizeof(float), cudaMemcpyHostToDevice));

  // 3. LAUNCH INITIAL SORT KERNEL
  // This creates the initial sorted chunks of size ELEMENTS_PER_BLOCK
  unsigned int numBlocks = (N + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
  initialSortKernel<<<numBlocks, BLOCK_SIZE>>>(d_src, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // 4. LAUNCH MERGE KERNEL IN A LOOP (Iterative Merging)
  for (unsigned int width = ELEMENTS_PER_BLOCK; width < N; width *= 2) {
    // Each pass merges sorted chunks of size `width` into sorted chunks of size
    // `2*width`. The `d_src` and `d_dst` pointers are swapped each pass
    // (ping-pong buffering).

    // This loop launches kernels to merge pairs of chunks.
    for (unsigned int i = 0; i < N; i += 2 * width) {
      if (i + width >= N) {
        // Only one run is left at the end, with no partner to merge with. It
        // is already sorted, but it must still be copied into the destination
        // buffer: after the swap below, d_dst becomes the source of the next
        // pass, and without this copy the run would be replaced by stale data
        // from an earlier pass.
        CUDA_CHECK(cudaMemcpy(d_dst + i, d_src + i, (N - i) * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        continue;
      }

      unsigned int m = width;
      // The second chunk is shorter than `width` when it reaches the end.
      unsigned int n = std::min(width, N - (i + width));

      unsigned int merge_size = m + n;
      unsigned int merge_num_blocks =
          (merge_size + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;

      // Launch kernel to merge chunks from SRC and write to DST
      mergeKernel<<<merge_num_blocks, BLOCK_SIZE>>>(
          d_src + i,          // Pointer to first chunk in source
          d_src + i + width,  // Pointer to second chunk in source
          d_dst + i,          // Output pointer in destination
          m,                  // Size of first chunk
          n                   // Size of second chunk
      );
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Swap pointers for the next pass (ping-pong)
    float *temp = d_src;
    d_src = d_dst;
    d_dst = temp;  // d_dst is now scratch space for the next iteration
  }

  // 5. Copy the final sorted data from device back to host
  // The final, sorted data is in d_src (due to the last swap)
  CUDA_CHECK(
      cudaMemcpy(h_data, d_src, N * sizeof(float), cudaMemcpyDeviceToHost));

  // 6. Free device memory
  CUDA_CHECK(cudaFree(d_src));
  CUDA_CHECK(cudaFree(d_dst));
}

// ===================================================================================
// Main Function
// ===================================================================================
int main(int argc, char **argv) {
  lab::Args args(argc, argv);
  // Neither size is a multiple of ELEMENTS_PER_BLOCK, and both produce
  // passes where the run count is odd, so the trailing-run copy is exercised.
  const unsigned int N = static_cast<unsigned int>(
      args.get_int("n", args.quick() ? 10 * 1024 + 17 : 1000000));

  lab::print_device();
  printf("Sorting %u elements...\n", N);

  // Random values from a small range, so the input has many duplicates and
  // the merge's tie handling is tested too.
  std::mt19937 gen(5);
  std::uniform_int_distribution<int> dist(0, 999);
  std::vector<float> h_data(N);
  for (float &value : h_data) {
    value = static_cast<float>(dist(gen));
  }
  std::vector<float> expected = h_data;
  std::sort(expected.begin(), expected.end());

  parallelMergeSort(h_data.data(), N);

  const bool pass = lab::check_equal("sorted output", h_data, expected);
  return lab::finish(pass);
}
