#include <cuda_runtime.h>
#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

#define BLOCK_SIZE 1024  // Number of threads per block
#define MAX_VALUE 100    // Maximum value for counting sort

// Error checking macro
#define CUDA_CHECK(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " at " << file
              << ":" << line << std::endl;
    if (abort) exit(code);
  }
}

template <typename T>
constexpr inline T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

__global__ void histogram_kernel(int *input, int *hist, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    atomicAdd(&hist[input[idx]], 1);
  }
}

__global__ void exclusive_scan_kernel(int *input, int *output, int N) {
  extern __shared__ int temp[];  // shared memory
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;

  // Load input into shared memory
  if (idx < N) {
    temp[tid] = input[idx];
  } else {
    temp[tid] = 0;
  }
  __syncthreads();

  // Exclusive scan (updating using a temp buffer)
  for (int offset = 1; offset < blockDim.x; offset *= 2) {
    int val = 0;
    if (tid >= offset) {
      val = temp[tid - offset];
    }
    __syncthreads();  // ensure all reads are done
    temp[tid] += val;
    __syncthreads();  // ensure all writes are done
  }

  // Convert to exclusive by shifting right
  if (idx < N) {
    if (tid == 0)
      output[idx] = 0;
    else
      output[idx] = temp[tid - 1];
  }
}

__global__ void placement_kernel(int *input, int *prefix, int *output, int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < N) {
    int element = input[tid];
    int place = atomicAdd(&prefix[element], 1);
    output[place] = element;
  }
}

void exclusive_scan_host(int *hist, int *prefix, int range) {
  prefix[0] = 0;
  for (int i = 1; i < range; ++i) {
    prefix[i] = prefix[i - 1] + hist[i - 1];
  }
}

int main() {
  // Example usage of the reduction function
  const int N = 1024;

  // Allocate device and host memory for input data.
  int *d_input, *h_input;
  CUDA_CHECK(cudaMalloc((void **)&d_input, N * sizeof(int)));
  h_input = (int *)malloc(N * sizeof(int));

  // Allocate device and host memory for output data.
  int *d_output_histogram, *h_output_histogram;
  CUDA_CHECK(cudaMalloc((void **)&d_output_histogram, MAX_VALUE * sizeof(int)));
  h_output_histogram = (int *)malloc(MAX_VALUE * sizeof(int));

  int *d_output_scan, *h_output_scan;
  CUDA_CHECK(cudaMalloc((void **)&d_output_scan, N * sizeof(int)));
  h_output_scan = (int *)malloc(N * sizeof(int));

  int *d_output_place, *h_output_place;
  CUDA_CHECK(cudaMalloc((void **)&d_output_place, N * sizeof(int)));
  h_output_place = (int *)malloc(N * sizeof(int));

  // Initialize input data with random numbers between 0 and 1.5
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(0, MAX_VALUE - 1);

  for (int i = 0; i < N; i++) {
    h_input[i] = dis(gen);
  }
  // Copy input data to device
  CUDA_CHECK(
      cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice));

  // Launch kernel
  histogram_kernel<<<1, BLOCK_SIZE>>>(d_input, d_output_histogram, N);
  CUDA_CHECK(cudaGetLastError());

  exclusive_scan_kernel<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(
      d_output_histogram, d_output_scan, N);
  CUDA_CHECK(cudaGetLastError());

  placement_kernel<<<1, BLOCK_SIZE>>>(d_input, d_output_scan, d_output_place,
                                      N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy result back to host
  CUDA_CHECK(cudaMemcpy(h_output_place, d_output_place, MAX_VALUE * sizeof(int),
                        cudaMemcpyDeviceToHost));

  // Sort on the host (for demonstration purposes)
  std::sort(h_input, h_input + N);
  // Check if results match

  for (int i = 0; i < 10; i++) {
    printf("Place[%d]: %d\n", i, h_output_place[i]);
    printf("h_input[%d]: %d\n", i, h_input[i]);
  }

  bool match = true;
  for (int i = 0; i < N; i++) {
    if (h_input[i] != h_output_place[i]) {
      match = false;
      printf("❌Mismatch at index %d: %d (CPU) vs %d (GPU)\n", i, h_input[i],
             h_output_place[i]);
      break;
    }
  }
  if (match) {
    printf("✅Results match!\n");
  } else {
    printf("❌Results do not match!\n");
  }

  // Free device and host memory
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output_histogram));
  CUDA_CHECK(cudaFree(d_output_scan));
  CUDA_CHECK(cudaFree(d_output_place));
  free(h_output_place);
  free(h_output_histogram);
  free(h_output_scan);
  free(h_input);

  return 0;
}