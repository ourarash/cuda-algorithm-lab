#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <vector>

/**
 * @brief Sorts a vector of positive doubles using the Geometric Bucketing
 * method (Epsilon Sort).
 * * This function creates and returns a new vector where for any two elements
 * at indices i < j, the condition vec[i] <= (1 + epsilon) * vec[j] holds. The
 * original vector is not modified. This revised version assumes all input
 * numbers are positive.
 * * @param vec The input vector of doubles to be sorted. Passed by const
 * reference.
 * @param epsilon The tolerance factor for the sort (e.g., 0.25 for 25%). Must
 * be > 0.
 * @return A new vector containing the epsilon-sorted elements.
 */
std::vector<double> epsilonSort(const std::vector<double>& vec,
                                double epsilon) {
  // --- 1. Handle Edge Cases ---
  // If the vector is empty or has only one element, it's already sorted.
  // Epsilon must be positive for the log base to be > 1.
  if (vec.size() <= 1 || epsilon <= 0) {
    return vec;  // Return a copy of the original vector.
  }

  // --- 2. Find the Minimum Value (min_val) ---
  // As per the requirement, we assume all numbers are positive.
  double min_val = *std::min_element(vec.begin(), vec.end());

  // --- 3. Create and Populate Geometric Buckets ---
  // We use a map to store buckets. The keys (bucket indices) are automatically
  // sorted, which is perfect for concatenation later. This also handles
  // non-contiguous and sparse bucket indices efficiently.
  std::map<int, std::vector<double>> buckets;

  // The base for the logarithm is (1 + epsilon).
  double log_base = 1.0 + epsilon;

  for (double x : vec) {
    // Calculate bucket index k = floor( log_{1+e}(x / min_val) )
    // We use the change of base formula: log_b(a) = log(a) / log(b)
    int k = static_cast<int>(
        std::floor(std::log(x / min_val) / std::log(log_base)));
    buckets[k].push_back(x);
  }

  // --- 4. Concatenate Buckets to Build the New Vector ---
  std::vector<double> sorted_vec;
  // Reserve space to avoid multiple reallocations, improving performance.
  sorted_vec.reserve(vec.size());

  // The map iterates through its keys in ascending order.
  for (auto const& [bucket_index, bucket_contents] : buckets) {
    // Insert all elements from the current bucket into the new vector.
    // The order of elements within the same bucket does not matter for the
    // epsilon-sort property, so we can just append them as they are.
    sorted_vec.insert(sorted_vec.end(), bucket_contents.begin(),
                      bucket_contents.end());
  }

  return sorted_vec;
}

// A functor (function object) to compute the bucket index for each element.
// This will be executed in parallel on the GPU for each element.
struct compute_bucket_functor {
  const double min_val;
  const double log_base;

  // Constructor to pass in the required values
  compute_bucket_functor(double _min_val, double _epsilon)
      : min_val(_min_val), log_base(1.0 + _epsilon) {}

  // The 'operator()' is what Thrust calls for each element.
  // It takes a value 'x' and returns its calculated bucket index.
  __host__ __device__ int operator()(const double& x) const {
    // Avoid log of zero or negative by handling the min_val case.
    if (x == min_val) {
      return 0;
    }
    // Calculate bucket index k = floor( log_{1+e}(x / min_val) )
    // Using change of base: log_b(a) = log(a) / log(b)
    return static_cast<int>(floor(log(x / min_val) / log(log_base)));
  }
};

/**
 * @brief Sorts a vector of positive doubles on the GPU using the Geometric
 * Bucketing method. This function uses a three-pass, sort-by-key approach with
 * the Thrust library. The original vector is modified in place.
 *
 * @param d_vec A Thrust device_vector of doubles to be sorted.
 * @param epsilon The tolerance factor for the sort (e.g., 0.25 for 25%).
 */
void epsilonSortGPU(thrust::device_vector<double>& d_vec, double epsilon) {
  // Handle edge cases
  if (d_vec.size() <= 1 || epsilon <= 0) {
    return;
  }

  // --- Pass 1: Compute min_val on GPU ---
  // Use Thrust's parallel reduction to find the minimum element.
  double min_val = *thrust::min_element(d_vec.begin(), d_vec.end());

  // --- Pass 2: Compute Bucket Indices in Parallel ---
  // Create a device_vector to store the bucket indices (the keys for our sort).
  thrust::device_vector<int> d_keys(d_vec.size());

  // Use thrust::transform to apply our functor to each element of d_vec
  // and store the resulting bucket index in d_keys.
  thrust::transform(d_vec.begin(), d_vec.end(), d_keys.begin(),
                    compute_bucket_functor(min_val, epsilon));

  // --- Pass 3: Sort by Bucket ID using Thrust ---
  // thrust::sort_by_key sorts the keys (d_keys) and applies the same
  // reordering to the values (d_vec). This groups all elements by their
  // bucket ID, achieving the epsilon-sort property.
  thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_vec.begin());
}

// Helper function to print a vector
void printVector(const std::string& title, const std::vector<double>& vec) {
  std::cout << title;
  for (double val : vec) {
    std::cout << val << " ";
  }
  std::cout << std::endl;
}

void printVector(const std::string& title,
                 const thrust::host_vector<double>& vec) {
  std::cout << title;
  for (double val : vec) {
    std::cout << val << " ";
  }
  std::cout << std::endl;
}

bool validateEpsilonSort(const thrust::host_vector<double>& vec,
                         double epsilon) {
  if (vec.size() <= 1) {
    return true;
  }

  double factor = 1.0 + epsilon;
  for (size_t i = 0; i < vec.size(); ++i) {
    for (size_t j = i + 1; j < vec.size(); ++j) {
      // Check for the defining inversion property of epsilon-sort.
      // If for i < j, we find vec[i] > (1+e)*vec[j], it's a violation.
      if (vec[i] > factor * vec[j]) {
        std::cerr << "Validation failed! At indices i=" << i << ", j=" << j
                  << std::endl;
        std::cerr << "vec[i] = " << vec[i] << ", vec[j] = " << vec[j]
                  << std::endl;
        std::cerr << "Condition failed: " << vec[i] << " > " << factor * vec[j]
                  << std::endl;
        return false;
      }
    }
  }
  return true;
}

int main() {
  // --- Example Usage ---
  std::vector<double> h_numbers = {1,  5,  4.8, 10, 9.5, 20, 4,  8,
                                   45, 15, 50,  6,  80,  7,  100};
  double epsilon = 0.25;

  printVector("Host: Original vector: ", h_numbers);

  // 1. Copy data from host (CPU) to device (GPU)
  thrust::device_vector<double> d_numbers = h_numbers;

  // 2. Call the GPU sorting function
  epsilonSortGPU(d_numbers, epsilon);

  // 3. Copy the results from device back to host to print them
  thrust::host_vector<double> h_sorted_numbers = d_numbers;

  printVector("Host: Epsilon-sorted vector: ", h_sorted_numbers);

  // 4. Validate the result on the CPU
  bool is_valid = validateEpsilonSort(h_sorted_numbers, epsilon);
  if (is_valid) {
    std::cout
        << "Validation successful: The vector is correctly epsilon-sorted."
        << std::endl;
  } else {
    std::cout
        << "Validation FAILED: The vector is NOT correctly epsilon-sorted."
        << std::endl;
  }

  return 0;
}
