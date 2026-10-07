/*
 * Kogge-Stone Inclusive Scan
 *
 * High-Level Algorithm:
 * This is the simplest shared-memory prefix scan in this folder. Each stage
 * doubles the distance that every thread can "see" to its left.
 *
 * Phase 1 (Load):
 * - One block loads the input into shared memory.
 *
 * Phase 2 (Recursive Doubling):
 * - At offset 1, thread i adds element i - 1.
 * - At offset 2, thread i adds element i - 2.
 * - At offset 4, thread i adds element i - 4.
 * - This continues until the offset covers the whole block.
 *
 * Why this version is useful:
 * - It is very easy to follow.
 * - It exposes the core scan idea clearly.
 *
 * Why it is not optimal:
 * - It does O(n log n) total work.
 * - It uses two barriers per stage because the scan is updated in place.
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
std::vector<double> cpu_inclusive_scan(const std::vector<float>& input) {
  std::vector<double> output(input.size());
  double running = 0.0;
  for (size_t i = 0; i < input.size(); ++i) {
    running += input[i];
    output[i] = running;
  }
  return output;
}

__global__ void kogge_stone_scan(float* output, const float* input, int n) {
  __shared__ float shared[kBlockSize];

  const int tid = threadIdx.x;
  shared[tid] = (tid < n) ? input[tid] : 0.0f;
  __syncthreads();

  for (int offset = 1; offset < n; offset *= 2) {
    float addend = 0.0f;
    if (tid >= offset && tid < n) {
      addend = shared[tid - offset];
    }

    // All reads for this stage must complete before any thread writes.
    __syncthreads();

    if (tid >= offset && tid < n) {
      shared[tid] += addend;
    }

    // The next stage must see a fully updated shared-memory snapshot.
    __syncthreads();
  }

  if (tid < n) {
    output[tid] = shared[tid];
  }
}

int main() {
  static_assert(kElementCount <= kBlockSize,
                "This demo uses a single block only.");

  lab::print_device();

  const std::vector<float> h_input =
      lab::random_uniform<float>(kElementCount, 1.0f, 1.1f, 7);
  std::vector<float> h_output(kElementCount);

  float* d_input = nullptr;
  float* d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, kElementCount * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, kElementCount * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), kElementCount * sizeof(float),
                        cudaMemcpyHostToDevice));

  kogge_stone_scan<<<1, kBlockSize>>>(d_output, d_input, kElementCount);
  CUDA_CHECK_LAUNCH();

  CUDA_CHECK(cudaMemcpy(h_output.data(), d_output,
                        kElementCount * sizeof(float), cudaMemcpyDeviceToHost));

  const std::vector<double> expected = cpu_inclusive_scan(h_input);
  printf("Last GPU output: %.6f\n", h_output.back());
  printf("Last CPU output: %.6f\n", expected.back());
  const bool pass = lab::check_close("scan", h_output, expected,
                                     /*rtol=*/1e-5, /*atol=*/1e-6);

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  return lab::finish(pass);
}
