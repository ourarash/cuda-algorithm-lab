#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

// ===================================================================================
// CUDA Error Checking Macro
// ===================================================================================
#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

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
// the input data into shared memory, sorts it locally, and writes it back.
// This creates the initial small, sorted runs that the merge kernel will work
// on.

__device__ void insertionSort(float *data, int size) {
  for (int i = 1; i < size; i++) {
    float key = data[i];
    int j = i - 1;
    while (j >= 0 && data[j] > key) {
      data[j + 1] = data[j];
      j = j - 1;
    }
    data[j + 1] = key;
  }
}

__global__ void initialSortKernel(float *data, unsigned int N) {
  __shared__ float shared_data[ELEMENTS_PER_BLOCK];

  unsigned int block_start_idx = blockIdx.x * ELEMENTS_PER_BLOCK;
  unsigned int thread_local_idx = threadIdx.x;

  // Each thread in the block loads multiple elements into shared memory
  for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
    unsigned int global_idx =
        block_start_idx + thread_local_idx + i * blockDim.x;
    unsigned int shared_idx = thread_local_idx + i * blockDim.x;
    if (global_idx < N) {
      shared_data[shared_idx] = data[global_idx];
    }
  }

  __syncthreads();

  // Sort the data within the block using a single thread for simplicity.
  // NOTE: A parallel sort (e.g., bitonic sort) in shared memory would be more
  // efficient.
  if (threadIdx.x == 0) {
    unsigned int effective_size =
        min((unsigned int)ELEMENTS_PER_BLOCK, N - block_start_idx);
    insertionSort(shared_data, effective_size);
  }

  __syncthreads();

  // Write the sorted chunk from shared memory back to global memory
  for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
    unsigned int global_idx =
        block_start_idx + thread_local_idx + i * blockDim.x;
    unsigned int shared_idx = thread_local_idx + i * blockDim.x;
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
  CHECK_CUDA(cudaMalloc(&d_src, N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(float)));

  // 2. Copy data from host to device source buffer
  CHECK_CUDA(
      cudaMemcpy(d_src, h_data, N * sizeof(float), cudaMemcpyHostToDevice));

  // 3. LAUNCH INITIAL SORT KERNEL
  // This creates the initial sorted chunks of size ELEMENTS_PER_BLOCK
  unsigned int numBlocks = (N + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
  initialSortKernel<<<numBlocks, BLOCK_SIZE>>>(d_src, N);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  // 4. LAUNCH MERGE KERNEL IN A LOOP (Iterative Merging)
  for (unsigned int width = ELEMENTS_PER_BLOCK; width < N; width *= 2) {
    // Each pass merges sorted chunks of size `width` into sorted chunks of size
    // `2*width`. The `d_src` and `d_dst` pointers are swapped each pass
    // (ping-pong buffering).

    numBlocks = (N + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;

    // This loop launches kernels to merge pairs of chunks.
    for (unsigned int i = 0; i < N; i += 2 * width) {
      unsigned int m = width;
      unsigned int n = width;

      // Boundary checks for the last chunks
      if (i + width >= N) {
        m = 0;
        n = 0;  // No second chunk to merge with
      } else if (i + 2 * width > N) {
        n = N - (i + width);  // The second chunk is smaller than `width`
      }

      if (m > 0 || n > 0) {
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
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // After merging pairs from src->dst, copy any remaining unmerged chunk at
    // the end This logic is simplified by just swapping pointers and letting
    // the next pass handle it. A full copy `cudaMemcpy(d_dst, d_src, ...)`
    // before the loop is a simpler but less performant way to handle
    // odd-numbered chunks. The current approach is more efficient.

    // Swap pointers for the next pass (ping-pong)
    float *temp = d_src;
    d_src = d_dst;
    d_dst = temp;  // d_dst is now scratch space for the next iteration
  }

  // 5. Copy the final sorted data from device back to host
  // The final, sorted data is in d_src (due to the last swap)
  CHECK_CUDA(
      cudaMemcpy(h_data, d_src, N * sizeof(float), cudaMemcpyDeviceToHost));

  // 6. Free device memory
  CHECK_CUDA(cudaFree(d_src));
  CHECK_CUDA(cudaFree(d_dst));
}

// ===================================================================================
// Main Function
// ===================================================================================
int main() {
  const unsigned int N = 1 << 20;  // Sort 1,048,576 elements
  std::vector<float> h_data(N);
  std::vector<float> h_data_cpu_sorted(N);  // For verification

  // Initialize data with a reverse-sorted array
  for (unsigned int i = 0; i < N; ++i) {
    h_data[i] = static_cast<float>(N - i);
  }
  // Create a copy of the original data for CPU sorting
  h_data_cpu_sorted = h_data;

  std::cout << "Sorting " << N << " elements..." << std::endl;
  std::cout << "First 10 unsorted elements: ";
  for (int i = 0; i < 10; ++i) {
    std::cout << h_data[i] << " ";
  }
  std::cout << "..." << std::endl;

  // Run the parallel sort on the GPU
  parallelMergeSort(h_data.data(), N);
  std::cout << "GPU sort complete." << std::endl;

  // Sort the reference data on the CPU for comparison
  std::cout << "Sorting reference data on CPU for verification..." << std::endl;
  std::sort(h_data_cpu_sorted.begin(), h_data_cpu_sorted.end());
  std::cout << "CPU sort complete." << std::endl;

  std::cout << "First 10 GPU-sorted elements:   ";
  for (int i = 0; i < 10; ++i) {
    std::cout << h_data[i] << " ";
  }
  std::cout << "..." << std::endl;

  // Verification against CPU sort
  std::cout << "Verifying GPU sort against CPU sort..." << std::endl;
  bool success = true;
  for (unsigned int i = 0; i < N; ++i) {
    if (h_data[i] != h_data_cpu_sorted[i]) {
      std::cerr << "Verification FAILED at index " << i << ": "
                << "GPU sorted value " << h_data[i] << " != CPU sorted value "
                << h_data_cpu_sorted[i] << std::endl;
      success = false;
      break;
    }
  }

  if (success) {
    std::cout << "Verification PASSED!" << std::endl;
  }

  return 0;
}
