/*
 * Hillis-Steele Inclusive Scan with Double Buffering
 *
 * High-Level Algorithm:
 * This version keeps the same recursive-doubling idea as Kogge-Stone, but each
 * stage reads from one shared-memory buffer and writes into another.
 *
 * Phase 1 (Load):
 * - Copy the input into the "ping" buffer.
 *
 * Phase 2 (Ping-Pong Scan Stages):
 * - Read the previous stage from one buffer.
 * - Write the next stage into the other buffer.
 * - Swap roles and repeat with a doubled offset.
 *
 * Why this version matters:
 * - The data flow is easier to reason about because every stage reads a stable
 *   snapshot from the previous stage.
 * - It avoids in-place read-after-write hazards inside a stage.
 *
 * Cost:
 * - It still performs O(n log n) work.
 * - It trades extra shared memory for cleaner staging.
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

__global__ void hillis_steele_scan(float* output, const float* input, int n) {
  __shared__ float buffers[2][kBlockSize];

  const int tid = threadIdx.x;
  buffers[0][tid] = (tid < n) ? input[tid] : 0.0f;
  __syncthreads();

  int current = 0;
  for (int offset = 1; offset < n; offset *= 2) {
    const int previous = current;
    current = 1 - current;

    float value = buffers[previous][tid];
    if (tid >= offset && tid < n) {
      value += buffers[previous][tid - offset];
    }
    buffers[current][tid] = value;

    // Every thread must finish writing this stage before the next stage reads.
    __syncthreads();
  }

  if (tid < n) {
    output[tid] = buffers[current][tid];
  }
}

int main() {
  static_assert(kElementCount <= kBlockSize,
                "This demo uses a single block only.");

  lab::print_device();

  const std::vector<float> h_input =
      lab::random_uniform<float>(kElementCount, 1.0f, 1.1f, 11);
  std::vector<float> h_output(kElementCount);

  float* d_input = nullptr;
  float* d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, kElementCount * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, kElementCount * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), kElementCount * sizeof(float),
                        cudaMemcpyHostToDevice));

  hillis_steele_scan<<<1, kBlockSize>>>(d_output, d_input, kElementCount);
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
