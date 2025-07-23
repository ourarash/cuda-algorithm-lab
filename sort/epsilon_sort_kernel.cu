#include <algorithm>
#include <cfloat>  // Added for DBL_MAX
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>  // Added for random number generation
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
__global__ void find_min_kernel(const double* d_in, double* d_out, size_t n) {
  extern __shared__ double s_data[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    s_data[tid] = d_in[i];
  } else {
    // CORRECTED: Use DBL_MAX, which is compatible with device code.
    s_data[tid] = DBL_MAX;
  }
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_data[tid] = fmin(s_data[tid], s_data[tid + s]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    d_out[blockIdx.x] = s_data[0];
  }
}

// Kernel 2: Calculate the size of each bucket (Histogram)
__global__ void histogram_kernel(const double* d_in, int* d_histogram, size_t n,
                                 double min_val, double log_of_base) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double x = d_in[i];
    int k = 0;
    if (x > min_val) {
      k = static_cast<int>(floor(log(x / min_val) / log_of_base));
    }
    atomicAdd(&d_histogram[k], 1);
  }
}

// Kernel 3: Exclusive Scan (Prefix Sum) to find bucket offsets
__global__ void exclusive_scan_kernel(const int* d_in, int* d_out,
                                      int num_buckets) {
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
                               size_t n, double min_val, double log_of_base) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double x = d_in[i];
    int k = 0;
    if (x > min_val) {
      k = static_cast<int>(floor(log(x / min_val) / log_of_base));
    }

    int local_offset = atomicAdd(&d_write_counters[k], 1);
    int final_pos = d_offsets[k] + local_offset;
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

  std::vector<double> h_min_partials(num_blocks_min);
  CUDA_CHECK(cudaMemcpy(h_min_partials.data(), d_min_partials,
                        num_blocks_min * sizeof(double),
                        cudaMemcpyDeviceToHost));
  double min_val = h_min_partials[0];
  for (size_t i = 1; i < h_min_partials.size(); ++i) {
    min_val = std::min(min_val, h_min_partials[i]);
  }

  // --- Pass 2: Histogram ---
  double max_val = *std::max_element(h_vec.begin(), h_vec.end());
  double log_of_base = log(1.0 + epsilon);
  int num_buckets =
      static_cast<int>(floor(log(max_val / min_val) / log_of_base)) + 2;

  CUDA_CHECK(cudaMalloc(&d_histogram, num_buckets * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_histogram, 0, num_buckets * sizeof(int)));

  const int num_blocks_hist = (n + threads_per_block - 1) / threads_per_block;
  histogram_kernel<<<num_blocks_hist, threads_per_block>>>(
      d_in, d_histogram, n, min_val, log_of_base);
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
      d_in, d_out, d_offsets, d_write_counters, n, min_val, log_of_base);
  CUDA_CHECK(cudaGetLastError());

  // --- Finalization ---
  CUDA_CHECK(cudaMemcpy(h_vec.data(), d_out, n * sizeof(double),
                        cudaMemcpyDeviceToHost));

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

// Note that this is O(N^2) and not efficient for large vectors..
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

/**
 * @brief Validates if a vector is correctly epsilon-sorted using an efficient
 * O(N) algorithm.
 *
 * @param vec The host vector to validate.
 * @param epsilon The epsilon value used for sorting.
 * @return true if the vector is epsilon-sorted, false otherwise.
 */
bool validateEpsilonSort_Optimized(const std::vector<double>& vec,
                                   double epsilon) {
  size_t n = vec.size();
  if (n <= 1) {
    return true;
  }

  // --- Pass 1: Compute suffix minimums ---
  std::vector<double> suffix_mins(n);
  suffix_mins[n - 1] = vec[n - 1];
  for (int i = n - 2; i >= 0; --i) {
    suffix_mins[i] = std::min(vec[i], suffix_mins[i + 1]);
  }

  // --- Pass 2: Validate using the pre-computed minimums ---
  double factor = 1.0 + epsilon;
  for (size_t i = 0; i < n - 1; ++i) {
    // Check vec[i] against the minimum of the entire rest of the array.
    if (vec[i] > factor * suffix_mins[i + 1]) {
      std::cerr << "Validation failed! At index i=" << i
                << ", vec[i] = " << vec[i] << std::endl;
      std::cerr << "The minimum value in the rest of the array is "
                << suffix_mins[i + 1] << std::endl;
      std::cerr << "Condition failed: " << vec[i] << " > "
                << factor * suffix_mins[i + 1] << std::endl;
      return false;
    }
  }

  return true;
}

void printVector(const std::string& title, const std::vector<double>& vec,
                 int limit = 10) {
  if (vec.empty()) {
    std::cout << title << " (empty vector)" << std::endl;
    return;
  }
  std::cout << title << " (size: " << vec.size() << "): ";
  int count = 0;
  for (double val : vec) {
    std::cout << val << " ";
    if (++count >= limit) break;
  }
  std::cout << std::endl;
}

int main() {
  const int N = 1024 * 1024;
  double epsilon = 0.25;

  // --- Generate Random Data ---
  std::vector<double> h_numbers(N);
  std::random_device rd;
  std::mt19937 gen(rd());
  // Generate random positive numbers, e.g., between 1.0 and 1000.0
  std::uniform_real_distribution<> distr(1.0, 1000.0);

  for (int i = 0; i < N; ++i) {
    h_numbers[i] = distr(gen);
  }

  std::cout << "Generated " << N << " random numbers." << std::endl;

  // --- Run Epsilon Sort ---
  epsilonSortGPU_Manual(h_numbers, epsilon);
  std::cout << "Epsilon sort completed." << std::endl;

  printVector("Sorted Numbers (First items)", h_numbers,
              /*limit*/ 20);

  printVector("Sorted Numbers (Last items)",
              std::vector<double>(h_numbers.rbegin(), h_numbers.rend()),
              /*limit*/ 20);

  // --- Validate the Result ---
  if (validateEpsilonSort_Optimized(h_numbers, epsilon)) {
    std::cout << "Validation successful!" << std::endl;
  } else {
    std::cout << "Validation FAILED!" << std::endl;
  }

  return 0;
}
