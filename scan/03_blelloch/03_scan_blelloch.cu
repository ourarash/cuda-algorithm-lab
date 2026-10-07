/*
 * Blelloch Exclusive Scan
 *
 * High-Level Algorithm:
 * Blelloch scan is a work-efficient tree scan. Unlike Kogge-Stone and
 * Hillis-Steele, it does O(n) total work.
 *
 * Phase 1 (Upsweep / Reduce):
 * - Build a sum tree in shared memory.
 * - Each level combines neighboring segments into a larger segment sum.
 *
 * Phase 2 (Root Initialization):
 * - Replace the root with zero.
 * - That zero is what turns the final result into an exclusive scan.
 *
 * Phase 3 (Downsweep):
 * - Traverse back down the tree.
 * - Each parent prefix is distributed to its children so every position
 *   receives the sum of all earlier elements.
 *
 * Important constraint:
 * - This textbook single-block implementation assumes n is a power of two.
 */
#include <cstdio>
#include <vector>

#include "lab.cuh"

constexpr int kBlockSize = 1024;
constexpr int kElementCount = 1024;

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

__global__ void blelloch_scan(float* output, const float* input, int n) {
  __shared__ float shared[kBlockSize];

  const int tid = threadIdx.x;
  shared[tid] = (tid < n) ? input[tid] : 0.0f;
  __syncthreads();

  for (int offset = 1; offset < n; offset *= 2) {
    int right = ((tid + 1) * offset * 2) - 1;
    if (right < n) {
      shared[right] += shared[right - offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    shared[n - 1] = 0.0f;
  }
  __syncthreads();

  for (int offset = n / 2; offset >= 1; offset /= 2) {
    int right = ((tid + 1) * offset * 2) - 1;
    int left = right - offset;

    if (right < n) {
      float left_value = shared[left];
      shared[left] = shared[right];
      shared[right] += left_value;
    }
    __syncthreads();
  }

  if (tid < n) {
    output[tid] = shared[tid];
  }
}

int main() {
  static_assert(kElementCount <= kBlockSize,
                "This demo uses a single block only.");
  static_assert((kElementCount & (kElementCount - 1)) == 0,
                "Blelloch scan requires a power-of-two problem size here.");

  lab::print_device();

  const std::vector<float> h_input =
      lab::random_uniform<float>(kElementCount, 1.0f, 1.1f, 23);
  std::vector<float> h_output(kElementCount);

  float* d_input = nullptr;
  float* d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, kElementCount * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, kElementCount * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), kElementCount * sizeof(float),
                        cudaMemcpyHostToDevice));

  blelloch_scan<<<1, kBlockSize>>>(d_output, d_input, kElementCount);
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
