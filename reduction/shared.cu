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

__global__ void reduction(float *input, float *partialSums, unsigned int N) {
  unsigned int tid = threadIdx.x;
  unsigned int i =
      blockIdx.x * blockDim.x * INPUT_SEGMENTS_PER_THREAD_BLOCK + tid;

  __shared__ float sharedData[BLOCK_SIZE];

  sharedData[tid] = 0.0f;
#pragma unroll
  // Each thread processes multiple segments of the input array.
  // This increases the amount of data processed per thread block,
  // improving memory coalescing and reducing the number of kernel launches.
  for (int j = 0; j < INPUT_SEGMENTS_PER_THREAD_BLOCK; ++j) {
    if (i + j * blockDim.x < N) {
      sharedData[tid] += input[i + j * blockDim.x];
    }
  }

  __syncthreads();

// Perform reduction within the block using a tree-based approach.
#pragma unroll
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      sharedData[tid] += sharedData[tid + stride];
    }
    __syncthreads();
  }

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
float KahanSummation(float *input, int n) {
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

float LaunchReduction(float *input, float *partialSums, int n) {
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
  CHECK_CUDA(cudaMalloc((void **)&d_input, N * sizeof(float)));
  h_input = (float *)malloc(N * sizeof(float));
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
  float *d_partialSums;
  float *h_partialSums = (float *)malloc(numBlocks * sizeof(float));
  CHECK_CUDA(cudaMalloc((void **)&d_partialSums, numBlocks * sizeof(float)));
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