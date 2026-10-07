/*
 * Blelloch Exclusive Scan with Bank-Conflict Padding
 *
 * High-Level Algorithm:
 * This kernel keeps the same Blelloch up-sweep / down-sweep structure, but it
 * changes how shared memory is indexed.
 *
 * Why padding helps:
 * - Shared memory is split into 32 banks.
 * - Tree scans often make neighboring threads access addresses that alias onto
 *   the same bank.
 * - Adding a small offset every 32 elements spreads those accesses across
 *   banks and reduces serialization.
 *
 * Important detail:
 * - Padding must be applied consistently on load, tree updates, root reset, and
 *   final store. The original file only applied it in part of the algorithm,
 *   which made the example incorrect.
 */
#include <cstdio>
#include <vector>

#include "lab.cuh"

constexpr int kBlockSize = 1024;
constexpr int kElementCount = 1024;
constexpr int kNumBanks = 32;
constexpr int kPadding = kBlockSize / kNumBanks;
constexpr int kPaddedSize = kBlockSize + kPadding;

__host__ __device__ constexpr int conflict_free_index(int index) {
  return index + (index / kNumBanks);
}

// CPU reference, accumulated in double precision. The GPU adds the same
// numbers in a different (tree) order, so a float reference would disagree
// with it by more than the GPU's own rounding error: one float ULP at 1024 is
// already 1.2e-4. Against the exact sums the GPU is accurate to ~3e-7.
std::vector<double> cpu_exclusive_scan(const std::vector<float>& input) {
  std::vector<double> output(input.size());
  double running = 0.0;
  for (size_t i = 0; i < input.size(); ++i) {
    output[i] = running;
    running += input[i];
  }
  return output;
}

__global__ void blelloch_scan_padded(float* output, const float* input, int n) {
  __shared__ float shared[kPaddedSize];

  const int tid = threadIdx.x;
  const int padded_tid = conflict_free_index(tid);
  shared[padded_tid] = (tid < n) ? input[tid] : 0.0f;
  __syncthreads();

  for (int offset = 1; offset < n; offset *= 2) {
    int right = ((tid + 1) * offset * 2) - 1;
    if (right < n) {
      int padded_right = conflict_free_index(right);
      int padded_left = conflict_free_index(right - offset);
      shared[padded_right] += shared[padded_left];
    }
    __syncthreads();
  }

  if (tid == 0) {
    shared[conflict_free_index(n - 1)] = 0.0f;
  }
  __syncthreads();

  for (int offset = n / 2; offset >= 1; offset /= 2) {
    int right = ((tid + 1) * offset * 2) - 1;
    int left = right - offset;

    if (right < n) {
      int padded_right = conflict_free_index(right);
      int padded_left = conflict_free_index(left);
      float left_value = shared[padded_left];
      shared[padded_left] = shared[padded_right];
      shared[padded_right] += left_value;
    }
    __syncthreads();
  }

  if (tid < n) {
    output[tid] = shared[padded_tid];
  }
}

int main() {
  static_assert(kElementCount <= kBlockSize,
                "This demo uses a single block only.");
  static_assert((kElementCount & (kElementCount - 1)) == 0,
                "Blelloch scan requires a power-of-two problem size here.");

  lab::print_device();

  const std::vector<float> h_input =
      lab::random_uniform<float>(kElementCount, 1.0f, 1.1f, 29);
  std::vector<float> h_output(kElementCount);

  float* d_input = nullptr;
  float* d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, kElementCount * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, kElementCount * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), kElementCount * sizeof(float),
                        cudaMemcpyHostToDevice));

  blelloch_scan_padded<<<1, kBlockSize>>>(d_output, d_input, kElementCount);
  CUDA_CHECK_LAUNCH();

  CUDA_CHECK(cudaMemcpy(h_output.data(), d_output,
                        kElementCount * sizeof(float), cudaMemcpyDeviceToHost));

  const std::vector<double> expected = cpu_exclusive_scan(h_input);
  printf("Last GPU output: %.6f\n", h_output.back());
  printf("Last CPU output: %.6f\n", expected.back());
  const bool pass = lab::check_close("scan", h_output, expected,
                                     /*rtol=*/1e-5, /*atol=*/1e-6);

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  return lab::finish(pass);
}
