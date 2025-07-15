#include <cuda_runtime.h>
#include <stdio.h>

#include <cmath>
#include <random>

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

// Check if two floating-point numbers are approximately equal by comparing
// their absolute difference to a small epsilon value, scaled by the maximum of
// their absolute values.
bool float_equal(float a, float b, float eps = 1e-5f) {
  return std::fabs(a - b) <= eps * std::fmax(std::fabs(a), std::fabs(b));
}

__global__ void reduction(float *input, float *partialSums, unsigned int N) {
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x * 2 + tid;

  __shared__ float sharedData[BLOCK_SIZE];

  float val1 = (i < N) ? input[i] : 0.0f;
  float val2 = (i + blockDim.x < N) ? input[i + blockDim.x] : 0.0f;
  sharedData[tid] = val1 + val2;
  __syncthreads();

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
  float cpu_result = CpuReduction(h_input, N);
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