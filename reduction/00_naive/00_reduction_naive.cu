/*
 * Naive Parallel Reduction
 * 
 * High-Level Algorithm:
 * This kernel implements a basic tree-based parallel reduction directly in 
 * global memory. The algorithm works by pairing up elements and continually 
 * halving the number of elements to process in each step until a single sum remains.
 * 
 * Tree Structure (Growing Stride):
 * - Step 1: Stride 1. Threads add elements 1 apart (e.g., T0 handles [0]+[1], T1 handles [2]+[3]).
 * - Step 2: Stride 2. Threads add elements 2 apart (e.g., T0 handles [0]+[2], T2 handles [4]+[6]).
 * - Step N: Stride doubles each iteration.
 * 
 * Drawbacks:
 * - Uses slow global memory for all intermediate steps.
 * - High Warp Divergence: The condition `threadIdx.x % stride == 0` leaves many threads 
 *   in a 32-thread warp inactive while a few do the work, wasting compute cycles.
 * - Uncoalesced Memory: As the stride grows, active threads access memory locations that 
 *   are far apart, breaking memory coalescing rules and tanking memory bandwidth.
 * 
 * Why not a shrinking stride here?
 * - We *could* use a shrinking stride (packed threads) in global memory to fix warp 
 *   divergence. However, this naive implementation uses a growing stride because it represents 
 *   the most direct conceptual translation of a binary tree. Even if we packed the threads, 
 *   it would still suffer from massive global memory latency, which is why the next real 
 *   optimization leap is to move to shared memory entirely.
 */
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256  // Number of threads per block

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

template <typename T>
constexpr inline T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

bool float_equal(float a, float b, float eps = 1e-5f) {
  return std::fabs(a - b) <= eps * std::fmax(std::fabs(a), std::fabs(b));
}


// -------------------------------------------------------------------------
// Naive Parallel Reduction Kernel
// -------------------------------------------------------------------------
// This is a basic form of parallel reduction without employing shared memory. 
// It works in-place directly on the global memory `input` array.
// Note: Each thread block processes 2 * BLOCK_SIZE elements.
// 
// Drawbacks of this naive approach:
// 1. Heavy reliance on slow global memory.
// 2. High warp divergence (e.g., `if (threadIdx.x % stride == 0)` causes only 
//    a fraction of threads in a warp to actually do work while others idle).
// 3. Uncoalesced memory accesses pattern in later iterations of the loop.
__global__ void reduction(float *input, float *partialSums, unsigned int N) {
  // Calculate global segment start for the current block.
  // Each block handles 2 * blockDim.x elements.
  unsigned int segment = blockIdx.x * blockDim.x * 2;
  
  // Calculate specific thread's starting index within the segment.
  // We multiply by 2 because each thread pair starts with a distance of 1 stride.
  unsigned int i = segment + threadIdx.x * 2;

  // Stride doubles each iteration for tree-based reduction: 1, 2, 4, 8, ...
  for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
    // Only active threads perform the addition. This causes high warp divergence.
    if (threadIdx.x % stride == 0) {
      input[i] += input[i + stride];
    }
    // Block-wide synchronization to ensure all threads finish a level of the tree
    // before moving to the next level.
    __syncthreads();
  }

  // The total sum for this block ends up in the first element processed by thread 0.
  // Write the partial sum from this block to global memory.
  if (threadIdx.x == 0) {
    partialSums[blockIdx.x] = input[i];
  }
}

// We use Kahan summation to reduce numerical errors in floating-point addition.
float CpuReduction(float *input, int n) {
  float sum = 0.0f;
  float c = 0.0f;  // A running compensation for lost low-order bits.

  for (int i = 0; i < n; i++) {
    float y = input[i] - c;  // Subtract the error from the last addition.
    float t = sum + y;       // Add the corrected value to the sum.
    c = (t - sum) - y;  // Calculate the new error. This recovers the part of
                        // 'y' that was lost.
    sum = t;            // Update the sum.
  }
  return sum;
}

float LaunchReduction(float *input, float *partialSums, int n) {
  int numBlocks = ceil_div(n, (2 * BLOCK_SIZE));

  float elapsed_time_ms = 0.0f;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // Launch the reduction kernel
  reduction<<<numBlocks, BLOCK_SIZE>>>(input, partialSums, n);

  // Check for errors in kernel launch
  CHECK_CUDA(cudaGetLastError());

  cudaEventRecord(stop, 0);

  // Synchronize to ensure all threads have completed
  CHECK_CUDA(cudaDeviceSynchronize());
  cudaEventElapsedTime(&elapsed_time_ms, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return elapsed_time_ms;
}

int main() {
  // Example usage of the reduction function
  const int N = 1024 * 1024;
  int numBlocks = ceil_div(N, (2 * BLOCK_SIZE));

  // Allocate device and host memory for input data.
  float *d_input, *h_input;
  CHECK_CUDA(cudaMalloc((void **)&d_input, N * sizeof(float)));
  h_input = (float *)malloc(N * sizeof(float));
  // Initialize input data
  for (int i = 0; i < N; i++) {
    h_input[i] = static_cast<float>(i + 1);
  }
  // Copy input data to device
  CHECK_CUDA(
      cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

  // Allocate host and device memory for partial sums
  float *d_partialSums;
  float *h_partialSums = (float *)malloc(numBlocks * sizeof(float));
  CHECK_CUDA(cudaMalloc((void **)&d_partialSums, numBlocks * sizeof(float)));

  // Launch reduction kernel
  float elapsed_time_ms = LaunchReduction(d_input, d_partialSums, N);

  // Copy result back to host
  CHECK_CUDA(cudaMemcpy(h_partialSums, d_partialSums, numBlocks * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // Reduce the partial sums on the host
  float finalSum = 0.0f;
  for (int i = 0; i < numBlocks; i++) {
    // printf("Partial sum %d: %f\n", i, h_partialSums[i]);
    finalSum += h_partialSums[i];
  }

  printf("Final reduction result: %f\n", finalSum);

  // Perform CPU reduction for verification
  float cpu_result = CpuReduction(h_input, N);
  printf("CPU Reduction result: %f\n", cpu_result);

  // Check if results match
  if (float_equal(finalSum, cpu_result)) {
    printf("✅Results match!\n");
  } else {
    printf("Results do not match!\n");
  }

  printf("Kernel time:  %8.2f ms\n", elapsed_time_ms);

  // Free device and host memory
  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_partialSums));
  free(h_partialSums);
  free(h_input);

  return 0;
}