/**
 * This program demonstrates the use of Thrust to perform set operations on two
 * integer sets A and B. It computes the XOR of the two
 * sets, which contains elements that are in either A or B but not in both.
 *
 * Assumption: We assume that the sets A and B do not contain duplicate values
 * and contain non-negative integers.
 *
 * Algorithm:
 * - Find the maximum value in both sets to determine the size of a frequency
 *   map.
 * -  Create a frequency map to count occurrences of each element in A and B:
 *   - 0: if it is not in A or B
 *   - 1: if it is in A ONLY or B ONLY (this is what we want for XOR)
 *   - 2: if it is in BOTH A and B
 * - Copy the the range of [0, 1, ..., max_val] to the output for each index in
 * the frequency map that has a value of 1.
 */

#include <algorithm>
#include <iostream>
#include <vector>

// Thrust library headers
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>

// Functor to increment the count in the 'present' array using atomic
// operations.
struct IncrementPresent {
  // Raw pointer to the device data of the 'present' vector.
  int *present_ptr;

  // Constructor to initialize the functor with the device pointer
  IncrementPresent(int *ptr) : present_ptr(ptr) {}

  __device__ void operator()(int value) {
    // Assuming we don't have duplicate values in the input sets, we can avoid
    // using atomic add here.
    present_ptr[value]++;
  }
};

// Functor to check if a given integer value is equal to 1.
struct IsOne {
  __host__ __device__ bool operator()(int val) { return val == 1; }
};

int main() {
  // Initialize two sets A and B as host vectors.
  // Note that (by definition) we assume these sets do not contain duplicate
  // values.
  std::vector<int> h_A = {1, 2, 3, 4, 5, 6, 7};
  std::vector<int> h_B = {1, 2, 3, 10, 11, 12};

  // Print the initial sets for verification.
  std::cout << "Set A: ";
  for (int x : h_A) {
    std::cout << x << " ";
  }
  std::cout << std::endl;

  std::cout << "Set B: ";
  for (int x : h_B) {
    std::cout << x << " ";
  }
  std::cout << std::endl;

  // Copy host vectors  to device vectors.
  thrust::device_vector<int> d_A = h_A;
  thrust::device_vector<int> d_B = h_B;

  // Find the maximum value in set A.
  int max_A = 0;
  if (!d_A.empty()) {
    max_A = thrust::reduce(d_A.begin(), d_A.end(), 0, thrust::maximum<int>());
  }

  // Find the maximum value in set B.
  int max_B = 0;
  if (!d_B.empty()) {
    max_B = thrust::reduce(d_B.begin(), d_B.end(), 0, thrust::maximum<int>());
  }

  // Determine the overall maximum value.
  int max_val = std::max(max_A, max_B);

  // Create a new device vector called 'present'.
  // Its size is 'max_val + 1' and all elements are initialized to 0.
  // This vector will act as a frequency map for numbers up to 'max_val'.
  // Each element value is:
  // 0: if it is not in A or B
  // 1: if it is in A ONLY or B ONLY (this is what we want for XOR)
  // 2: if it is in BOTH A and B
  thrust::device_vector<int> present(max_val + 1, 0);

  // Get a raw pointer to the device data of the 'present' vector.
  // This raw pointer is passed to the IncrementPresent functor, allowing
  // direct modification of 'present' elements from within device kernels.
  int *present_ptr = thrust::raw_pointer_cast(present.data());

  // Iterate through each item in set A and increment its count in the 'present'
  // vector.
  thrust::for_each(thrust::device, d_A.begin(), d_A.end(),
                   IncrementPresent(present_ptr));

  // Iterate through each item in set B and increment its count in the 'present'
  // vector.
  thrust::for_each(thrust::device, d_B.begin(), d_B.end(),
                   IncrementPresent(present_ptr));

  // Create the output vector which holds the result of the XOR operation.
  size_t output_size =
      thrust::count_if(thrust::device, present.begin(), present.end(), IsOne());

  thrust::device_vector<int> d_xor(output_size);

  // Copy elements from the range [0, 1, ..., max_val] to the output if the
  // predicate on the stencil (i.e. the `present` vector) is true.
  thrust::copy_if(
      thrust::device,
      // The input to copy is the range of [0, 1, ..., max_val], both inclusive.
      // We use thrust::make_counting_iterator to generate this range.
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(static_cast<int>(max_val + 1)),
      // Stencil (secondary input): 'present'.
      present.begin(),
      // Output iterator: beginning of the resized vector
      d_xor.begin(),
      // Predicate: checks if the stencil value is 1.
      IsOne());

  // Copy the final result back to a host vector.
  thrust::host_vector<int> h_xor = d_xor;

  // Print the output.
  std::cout << "Result (A XOR B): ";
  if (h_xor.empty()) {
    std::cout << "Empty set." << std::endl;
  } else {
    for (int x : h_xor) {
      std::cout << x << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}
