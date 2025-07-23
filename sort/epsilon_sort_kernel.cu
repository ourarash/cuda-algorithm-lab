#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

// ==========================================================================
// CUDA Error Checking Utility
// ==========================================================================
#define CUDA_CHECK(err)                                                 \
  {                                                                     \
    cudaError_t err_ = (err);                                           \
    if (err_ != cudaSuccess) {                                          \
      std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ \
                << ": " << cudaGetErrorString(err_) << std::endl;       \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

// ==========================================================================
// CUDA Kernels for Epsilon Sort
// ==========================================================================

// Kernel 1: Parallel Reduction to find the minimum value
// This is a simple version; for extreme performance, a more complex
// shared-memory reduction would be used. This version uses a single block for
// simplicity.
__global__ void find_min_kernel(const double* d_in, double* d_out, size_t n) {
  extern __shared__ double s_data[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Load data into shared memory
  if (i < n) {
    s_data[tid] = d_in[i];
  } else {
    s_data[tid] = std::numeric_limits<double>::max();
  }
  __syncthreads();

  // Perform reduction in shared memory
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_data[tid] = fmin(s_data[tid], s_data[tid + s]);
    }
    __syncthreads();
  }

  // Write result for this block to global memory
  if (tid == 0) {
    d_out[blockIdx.x] = s_data[0];
  }
}

// Kernel 2: Calculate the size of each bucket (Histogram)
__global__ void histogram_kernel(const double* d_in, int* d_histogram, size_t n,
                                 double min_val, double log_base) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double x = d_in[i];
    int k = 0;
    if (x > min_val) {
      k = static_cast<int>(floor(log(x / min_val) / log_base));
    }
    atomicAdd(&d_histogram[k], 1);
  }
}

// Kernel 3: Exclusive Scan (Prefix Sum) to find bucket offsets
// This is a simple, sequential scan kernel suitable for a small number of
// buckets. A high-performance parallel scan would be used for a very large
// number of buckets.
__global__ void exclusive_scan_kernel(const int* d_in, int* d_out,
                                      int num_buckets) {
  // This kernel runs with a single thread as it's sequential.
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    d_out[0] = 0;
    for (int i = 1; i < num_buckets; ++i) {
      d_out[i] = d_out[i - 1] + d_in[i - 1];
    }
  }
}

// Kernel 4: Scatter elements to their final positions
__global__ void scatter_kernel(const double* d_in, double* d_out,
                               const int* d_offsets, int* d_write_counters,
                               size_t n, double min_val, double log_base) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double x = d_in[i];
    int k = 0;
    if (x > min_val) {
      k = static_cast<int>(floor(log(x / min_val) / log_base));
    }

    // Atomically get the write position within the bucket
    int local_offset = atomicAdd(&d_write_counters[k], 1);

    // Calculate final destination
    int final_pos = d_offsets[k] + local_offset;

    // Write the value
    d_out[final_pos] = x;
  }
}

// ==========================================================================
// Host Function to Orchestrate Epsilon Sort
// ==========================================================================

void epsilonSortGPU_Manual(std::vector<double>& h_vec, double epsilon) {
  size_t n = h_vec.size();
  if (n <= 1 || epsilon <= 0) return;

  // --- Device Memory Allocation ---
  double *d_in, *d_out, *d_min_partials;
  int *d_histogram, *d_offsets, *d_write_counters;

  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(double)));

  // Copy input data to device
  CUDA_CHECK(cudaMemcpy(d_in, h_vec.data(), n * sizeof(double),
                        cudaMemcpyHostToDevice));

  // --- Pass 1: Find Minimum Value ---
  const int threads_per_block = 256;
  const int num_blocks_min = (n + threads_per_block - 1) / threads_per_block;
  CUDA_CHECK(cudaMalloc(&d_min_partials, num_blocks_min * sizeof(double)));
  find_min_kernel<<<num_blocks_min, threads_per_block,
                    threads_per_block * sizeof(double)>>>(d_in, d_min_partials,
                                                          n);
  CUDA_CHECK(cudaGetLastError());

  // If there was more than one block, we'd need a second reduction pass.
  // For simplicity, we copy the partial minimums back and finish on the host.
  std::vector<double> h_min_partials(num_blocks_min);
  CUDA_CHECK(cudaMemcpy(h_min_partials.data(), d_min_partials,
                        num_blocks_min * sizeof(double),
                        cudaMemcpyDeviceToHost));
  double min_val = h_min_partials[0];
  for (size_t i = 1; i < h_min_partials.size(); ++i) {
    min_val = std::min(min_val, h_min_partials[i]);
  }

  // --- Pass 2: Histogram ---
  // Determine a reasonable max number of buckets. This depends on the data
  // range. For a general case, this needs careful thought. We'll estimate.
  double max_val = *std::max_element(h_vec.begin(), h_vec.end());
  double log_base = 1.0 + epsilon;
  int num_buckets =
      static_cast<int>(floor(log(max_val / min_val) / log(log_base))) + 2;

  CUDA_CHECK(cudaMalloc(&d_histogram, num_buckets * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_histogram, 0, num_buckets * sizeof(int)));

  const int num_blocks_hist = (n + threads_per_block - 1) / threads_per_block;
  histogram_kernel<<<num_blocks_hist, threads_per_block>>>(d_in, d_histogram, n,
                                                           min_val, log_base);
  CUDA_CHECK(cudaGetLastError());

  // --- Pass 3: Exclusive Scan ---
  CUDA_CHECK(cudaMalloc(&d_offsets, num_buckets * sizeof(int)));
  exclusive_scan_kernel<<<1, 1>>>(d_histogram, d_offsets, num_buckets);
  CUDA_CHECK(cudaGetLastError());

  // --- Pass 4: Scatter ---
  CUDA_CHECK(cudaMalloc(&d_write_counters, num_buckets * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_write_counters, 0, num_buckets * sizeof(int)));

  const int num_blocks_scatter =
      (n + threads_per_block - 1) / threads_per_block;
  scatter_kernel<<<num_blocks_scatter, threads_per_block>>>(
      d_in, d_out, d_offsets, d_write_counters, n, min_val, log_base);
  CUDA_CHECK(cudaGetLastError());

  // --- Finalization ---
  // Copy result back to host
  CUDA_CHECK(cudaMemcpy(h_vec.data(), d_out, n * sizeof(double),
                        cudaMemcpyDeviceToHost));

  // Free all device memory
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_min_partials));
  CUDA_CHECK(cudaFree(d_histogram));
  CUDA_CHECK(cudaFree(d_offsets));
  CUDA_CHECK(cudaFree(d_write_counters));
}

// ==========================================================================
// Host Validation and Main Function
// ==========================================================================

bool validateEpsilonSort(const std::vector<double>& vec, double epsilon) {
  if (vec.size() <= 1) return true;
  double factor = 1.0 + epsilon;
  for (size_t i = 0; i < vec.size(); ++i) {
    for (size_t j = i + 1; j < vec.size(); ++j) {
      if (vec[i] > factor * vec[j]) {
        std::cerr << "Validation failed! At indices i=" << i << ", j=" << j
                  << std::endl;
        std::cerr << "vec[i] = " << vec[i] << ", vec[j] = " << vec[j]
                  << std::endl;
        return false;
      }
    }
  }
  return true;
}

void printVector(const std::string& title, const std::vector<double>& vec) {
  std::cout << title;
  for (double val : vec) {
    std::cout << val << " ";
  }
  std::cout << std::endl;
}

int main() {
  std::vector<double> h_numbers = {1,  5,  4.8, 10, 9.5, 20, 4,  8,
                                   45, 15, 50,  6,  80,  7,  100};
  double epsilon = 0.25;

  printVector("Original vector:    ", h_numbers);

  epsilonSortGPU_Manual(h_numbers, epsilon);

  printVector("Epsilon-sorted vector:", h_numbers);

  if (validateEpsilonSort(h_numbers, epsilon)) {
    std::cout << "Validation successful!" << std::endl;
  } else {
    std::cout << "Validation FAILED!" << std::endl;
  }

  return 0;
}
