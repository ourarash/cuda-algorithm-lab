#include <cuda_runtime.h>
#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

#define BLOCK_SIZE 1024  // Number of threads per block
#define MAX_VALUE 100    // Maximum value for the input numbers

// Error checking macro for CUDA calls
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

/**
 * @brief Utility function to calculate ceiling division.
 * Used to determine the number of blocks needed for a kernel launch.
 */
template <typename T>
constexpr inline T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

/**
 * @brief GPU Kernel: Calculates a histogram of the input data.
 * Each thread atomically increments the bin corresponding to its input value.
 * @param input The input array of integers on the device.
 * @param hist The output histogram array on the device (size: MAX_VALUE).
 * @param N The total number of elements in the input array.
 */
__global__ void histogram_kernel(int *input, int *hist, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    // Atomically increment the count for the value found at input[idx]
    atomicAdd(&hist[input[idx]], 1);
  }
}

/**
 * @brief GPU Kernel: Places elements into their final sorted positions.
 * It uses the prefix sum array to determine the correct output index.
 * @param input The original unsorted input array on the device.
 * @param prefix The prefix sum (exclusive scan) of the histogram.
 * @param output The final sorted output array.
 * @param N The total number of elements in the input array.
 */
__global__ void placement_kernel(int *input, int *prefix, int *output, int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < N) {
    int element = input[tid];
    // Atomically get the correct position from the prefix sum array
    // and increment it for the next thread that has the same element value.
    int place = atomicAdd(&prefix[element], 1);
    output[place] = element;
  }
}

/**
 * @brief CPU Function: Performs an exclusive scan on the histogram.
 * This computes the starting index for each value in the final sorted array.
 * @param hist The input histogram array.
 * @param prefix The output array for the exclusive scan.
 * @param range The size of the histogram and prefix sum arrays (MAX_VALUE).
 */
void exclusive_scan_host(int *hist, int *prefix, int range) {
  prefix[0] = 0;
  for (int i = 1; i < range; ++i) {
    prefix[i] = prefix[i - 1] + hist[i - 1];
  }
}

int main() {
  const int N = 1024 * 16;  // Using a larger N for more robust testing
  const int grid_size = ceil_div(N, BLOCK_SIZE);

  // --- 1. Allocate Memory ---
  // Allocate all necessary memory on both the host (CPU) and device (GPU)
  int *h_input = (int *)malloc(N * sizeof(int));
  int *h_output_place = (int *)malloc(N * sizeof(int));
  int *h_histogram = (int *)malloc(MAX_VALUE * sizeof(int));
  int *h_prefix_sum = (int *)malloc(MAX_VALUE * sizeof(int));

  int *d_input, *d_output_place, *d_histogram, *d_prefix_sum;
  CUDA_CHECK(cudaMalloc((void **)&d_input, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&d_output_place, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&d_histogram, MAX_VALUE * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&d_prefix_sum, MAX_VALUE * sizeof(int)));

  // --- 2. Initialize Data ---
  // Create random input data on the host
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(0, MAX_VALUE - 1);
  for (int i = 0; i < N; i++) {
    h_input[i] = dis(gen);
  }

  // Copy the input data from host to device
  CUDA_CHECK(
      cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice));

  // --- 3. GPU Histogram Calculation ---
  // First, ensure the histogram memory on the device is zeroed out
  CUDA_CHECK(cudaMemset(d_histogram, 0, MAX_VALUE * sizeof(int)));
  histogram_kernel<<<grid_size, BLOCK_SIZE>>>(d_input, d_histogram, N);
  CUDA_CHECK(cudaGetLastError());

  // --- 4. CPU Exclusive Scan ---
  // Copy the histogram from the device to the host to perform the scan
  CUDA_CHECK(cudaMemcpy(h_histogram, d_histogram, MAX_VALUE * sizeof(int),
                        cudaMemcpyDeviceToHost));

  // Perform the scan on the host using the simple, correct serial function
  exclusive_scan_host(h_histogram, h_prefix_sum, MAX_VALUE);

  // Copy the resulting prefix sum array back to the device for the placement
  // step
  CUDA_CHECK(cudaMemcpy(d_prefix_sum, h_prefix_sum, MAX_VALUE * sizeof(int),
                        cudaMemcpyHostToDevice));

  // --- 5. GPU Placement ---
  // Use the prefix sum to place elements into their final sorted positions
  placement_kernel<<<grid_size, BLOCK_SIZE>>>(d_input, d_prefix_sum,
                                              d_output_place, N);
  CUDA_CHECK(cudaGetLastError());

  // Block until the device has completed all preceding tasks
  CUDA_CHECK(cudaDeviceSynchronize());

  // --- 6. Verification ---
  // Copy the final sorted result from the device back to the host
  CUDA_CHECK(cudaMemcpy(h_output_place, d_output_place, N * sizeof(int),
                        cudaMemcpyDeviceToHost));

  // Sort the original host input array to serve as the ground truth
  std::sort(h_input, h_input + N);

  // Compare the CPU-sorted result with the GPU-sorted result
  bool match = true;
  for (int i = 0; i < N; i++) {
    if (h_input[i] != h_output_place[i]) {
      match = false;
      printf("❌ Mismatch at index %d: %d (CPU) vs %d (GPU)\n", i, h_input[i],
             h_output_place[i]);
      break;
    }
  }
  if (match) {
    printf("✅ Results match!\n");
  } else {
    printf("❌ Results do not match!\n");
  }

  // --- 7. Free Memory ---
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output_place));
  CUDA_CHECK(cudaFree(d_histogram));
  CUDA_CHECK(cudaFree(d_prefix_sum));
  free(h_input);
  free(h_output_place);
  free(h_histogram);
  free(h_prefix_sum);

  return 0;
}
