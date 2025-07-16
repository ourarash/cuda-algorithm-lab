#include <cuda_runtime.h>
#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <random>

#define BLOCK_SIZE 256  // Number of threads per block
#define MAX_VALUE 1023  // Maximum value for counting sort

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

__global__ void histogram_kernel(int *input, int *hist, int N) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    atomicAdd(&hist[input[idx]], 1);
  }
}

void exclusive_scan_host(int *hist, int *prefix, int range) {
  prefix[0] = 0;
  for (int i = 1; i < range; ++i) {
    prefix[i] = prefix[i - 1] + hist[i - 1];
  }
}

float LaunchReduction(float *input, float *output, int n) {
  int numBlocks = ceil_div(n, (BLOCK_SIZE));

  float elapsed_time_ms = 0.0f;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // Launch the reduction kernel
  sort<<<numBlocks, BLOCK_SIZE>>>(input, output, n);

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

  // Allocate device and host memory for input data.
  float *d_input, *h_input;
  CHECK_CUDA(cudaMalloc((void **)&d_input, N * sizeof(float)));
  h_input = (float *)malloc(N * sizeof(float));

  // Allocate device and host memory for output data.
  float *d_output, *h_output;
  CHECK_CUDA(cudaMalloc((void **)&d_output, N * sizeof(float)));
  h_output = (float *)malloc(N * sizeof(float));

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

  // Launch kernel
  float elapsed_time_ms = LaunchReduction(d_input, d_output, N);

  // Copy result back to host
  CHECK_CUDA(cudaMemcpy(h_output, d_output, N * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // Sort on the host (for demonstration purposes)
  std::sort(h_input, h_input + N);
  // Check if results match
  bool match = true;
  for (int i = 0; i < N; i++) {
    if (std::fabs(h_input[i] - h_output[i]) > 1e-5f) {
      match = false;
      printf("❌Mismatch at index %d: %f (CPU) vs %f (GPU)\n", i, h_input[i],
             h_output[i]);
      break;
    }
  }
  if (match) {
    printf("✅Results match!\n");
  } else {
    printf("❌Results do not match!\n");
  }

  printf("Kernel time:  %8.2f ms\n", elapsed_time_ms);

  // Free device and host memory
  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_output));
  free(h_output);
  free(h_input);

  return 0;
}