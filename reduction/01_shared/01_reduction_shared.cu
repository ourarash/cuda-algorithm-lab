/*
 * Shared Memory Parallel Reduction (Optimized)
 * 
 * High-Level Algorithm:
 * This kernel optimizes the naive reduction by employing fast on-chip shared memory, 
 * sequential thread mapping, and a redesigned reduction tree to eliminate warp divergence.
 * 
 * Phase 1 (Global-to-Shared Accumulation):
 * Each thread sequentially reads multiple elements from global memory, accumulating them 
 * into a local register sum. This guarantees perfectly coalesced memory reads and increases 
 * the work-per-thread, amortizing overhead. The thread's partial sum is then written 
 * into shared memory.
 * 
 * Phase 2 (Tree Reduction in Shared Memory):
 * How this tree drastically differs from the Naive approach:
 * - Shrinking Stride vs. Growing Stride: The naive tree started with stride=1 and grew (1, 2, 4...). 
 *   This optimized tree starts at half the block size and shrinks by half (e.g., 128, 64, 32...).
 * - Packed Active Threads vs. Scattered: In the naive tree, active threads were scattered 
 *   (`tid % stride == 0`), destroying warp utilization. Here, the check is `tid < stride`. 
 *   This tightly packs all active threads contiguous to each other on the left side of the block.
 * - Resulting Hardware Efficiency: Because active threads are grouped perfectly together (0 to stride - 1), 
 *   entire warps will be 100% strictly active or 100% gracefully idle, successfully completely 
 *   eliminating warp divergence for all steps until the total active thread count drops below 32.
 * 
 * Why not a *growing* stride with packed threads in Shared Memory?
 * - If we packed threads but used a growing stride (e.g., `sharedData[2 * stride * tid] += ...`), 
 *   adjacent threads (T0, T1, T2) would access memory at leaps of 2, 4, 8, etc. Since shared 
 *   memory is divided into 32 banks, a stride of 2 causes 2-way bank conflicts (hardware serializes 
 *   the memory reads). A shrinking stride (`sharedData[tid] += ...`) ensures adjacent threads access 
 *   adjacent indices (stride of 1), completely eliminating both warp divergence AND bank conflicts.
 */
#include <cuda_runtime.h>
#include <stdio.h>

#include <cmath>
#include <random>

#define BLOCK_SIZE 256  // Number of threads per block
// Number of segments per thread block, i.e., how many input segments each
// thread block will process.
#define INPUT_SEGMENTS_PER_THREAD_BLOCK 4

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

// Check if two floating-point numbers are approximately equal by comparing
// their absolute difference to a small epsilon value, scaled by the maximum of
// their absolute values.
bool float_equal(float a, float b, float eps = 1e-5f) {
  return std::fabs(a - b) <= eps * std::fmax(std::fabs(a), std::fabs(b));
}

// -------------------------------------------------------------------------
// Shared Memory Parallel Reduction Kernel
// -------------------------------------------------------------------------
// This reduction kernel improves on the naive approach by utilizing fast
// on-chip shared memory, contiguous memory accesses, and reducing warp
// divergence. Instead of modifying global memory, the threads first load data
// and accumulate an initial sum into shared memory, then perform a tree
// reduction.
__global__ void reduction(float* input, float* partialSums, unsigned int N) {
  // Local thread ID within the block
  unsigned int tid = threadIdx.x;

  // Global starting index for this thread. Note that each thread block
  // handles exactly INPUT_SEGMENTS_PER_THREAD_BLOCK elements chunked by block
  // dimension.
  unsigned int i =
      blockIdx.x * blockDim.x * INPUT_SEGMENTS_PER_THREAD_BLOCK + tid;

  // Allocate shared memory for this block. Max size equals the number of
  // threads.
  __shared__ float sharedData[BLOCK_SIZE];

  // Initialize the shared memory element for this thread
  sharedData[tid] = 0.0f;

#pragma unroll
  // Phase 1: Global-to-Shared Accumulation
  // Each thread processes multiple segments of the input array sequentially.
  // This increases work-per-thread (amortizing instruction overhead),
  // guarantees coalesced global memory reads, and reduces total kernel
  // launches.
  for (int j = 0; j < INPUT_SEGMENTS_PER_THREAD_BLOCK; ++j) {
    if (i + j * blockDim.x < N) {
      sharedData[tid] += input[i + j * blockDim.x];
    }
  }

  // Ensure all threads have finished writing their initial sums to shared
  // memory.
  __syncthreads();

// Phase 2: Tree-based Reduction in Shared Memory
// We use a shrinking stride (stride /= 2) instead of a growing stride.
// This avoids warp divergence because active threads share the same consecutive
// warps (e.g., in the first iteration, threads 0 to 127 are active, spanning
// warps 0-3 fully).
#pragma unroll
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      // Add the value from the right half to the left half in shared memory.
      sharedData[tid] += sharedData[tid + stride];
    }
    // Block-wide synchronization loop is required at each depth of the tree.
    __syncthreads();
  }

  // At the end of the loop, sharedData[0] contains the sum for the entire
  // block. Thread 0 writes this value to the output array holding partial sums.
  if (tid == 0) {
    partialSums[blockIdx.x] = sharedData[0];
  }
}

/**
 * @brief Accurately sums an array of floating-point numbers using the Kahan
 * summation algorithm.
 * * This method minimizes floating-point error by tracking a running
 * compensation for the low-order bits that are lost during addition.
 * * @param input An array of floats to be summed.
 * @param n The number of elements in the array.
 * @return The highly accurate sum of the elements.
 */
float KahanSummation(float* input, int n) {
  // The main accumulator, which holds the running total.
  // Prone to precision errors when adding small numbers.
  float runningSum = 0.0f;

  // Stores the accumulated error from previous additions.
  // This is the "lost" part that we'll re-incorporate later.
  float errorCorrection = 0.0f;

  for (int i = 0; i < n; i++) {
    // 1. Compensate: Adjust the next input value by subtracting the error
    //    that was calculated in the previous iteration.
    float compensatedInput = input[i] - errorCorrection;

    // 2. Add: Add the compensated value to our main running sum.
    //    This is where precision can be lost.
    float newSum = runningSum + compensatedInput;

    // 3. Capture Error: Calculate the new error. This is the crucial step.
    //    It recovers the part of 'compensatedInput' that was lost when adding
    //    to 'runningSum'.
    errorCorrection = (newSum - runningSum) - compensatedInput;

    // 4. Update: Set the running sum to our new, albeit imprecise, total.
    //    The error has been safely stored for the next loop.
    runningSum = newSum;
  }

  return runningSum;
}

float LaunchReduction(float* input, float* partialSums, int n) {
  int numBlocks = ceil_div(n, (INPUT_SEGMENTS_PER_THREAD_BLOCK * BLOCK_SIZE));

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
  int numBlocks = ceil_div(N, (INPUT_SEGMENTS_PER_THREAD_BLOCK * BLOCK_SIZE));

  // Allocate device and host memory for input data.
  float *d_input, *h_input;
  CHECK_CUDA(cudaMalloc((void**)&d_input, N * sizeof(float)));
  h_input = (float*)malloc(N * sizeof(float));
  // Initialize input data with random numbers between 0 and 1.5
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(1.0f, 1.1f);

  for (int i = 0; i < N; i++) {
    h_input[i] = dis(gen);
  }
  // Copy input data to device
  CHECK_CUDA(
      cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

  // Allocate host and device memory for partial sums
  float* d_partialSums;
  float* h_partialSums = (float*)malloc(numBlocks * sizeof(float));
  CHECK_CUDA(cudaMalloc((void**)&d_partialSums, numBlocks * sizeof(float)));
  CHECK_CUDA(cudaMemset(d_partialSums, 0, numBlocks * sizeof(float)));

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
  float cpu_result = KahanSummation(h_input, N);
  printf("CPU Reduction result: %f\n", cpu_result);

  // Check if results match
  if (float_equal(finalSum, cpu_result)) {
    printf("✅Results match!\n");
  } else {
    printf("❌Results do not match!\n");
  }

  printf("Kernel time:  %8.2f ms\n", elapsed_time_ms);

  // Free device and host memory
  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_partialSums));
  free(h_partialSums);
  free(h_input);

  return 0;
}