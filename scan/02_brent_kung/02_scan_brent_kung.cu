/*
 * Brent-Kung Inclusive Scan
 *
 * High-Level Algorithm:
 * Brent-Kung is a classic compromise between shallow-depth scan networks like
 * Kogge-Stone and fully work-efficient tree scans like Blelloch.
 *
 * Phase 1 (Reduction / Upsweep):
 * - Threads build partial sums at the right edges of progressively larger
 *   segments.
 *
 * Phase 2 (Distribution):
 * - Those partial sums are pushed back down the tree so that the missing prefix
 *   information reaches the interior nodes.
 *
 * Why this version matters:
 * - It performs less work than Kogge-Stone/Hillis-Steele.
 * - It is a famous scan network that was missing from the original folder.
 * - It still produces an inclusive scan directly.
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

__global__ void brent_kung_scan(float* output, const float* input, int n) {
  __shared__ float shared[kBlockSize];

  const int tid = threadIdx.x;
  shared[tid] = (tid < n) ? input[tid] : 0.0f;
  __syncthreads();

  // Build partial sums on the right edge of each segment.
  for (int stride = 1; stride < n; stride *= 2) {
    int index = ((tid + 1) * stride * 2) - 1;
    if (index < n) {
      shared[index] += shared[index - stride];
    }
    __syncthreads();
  }

  // Push prefix information back down into the interior nodes.
  for (int stride = n / 4; stride > 0; stride /= 2) {
    int index = ((tid + 1) * stride * 2) - 1;
    if (index + stride < n) {
      shared[index + stride] += shared[index];
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

  lab::print_device();

  const std::vector<float> h_input =
      lab::random_uniform<float>(kElementCount, 1.0f, 1.1f, 19);
  std::vector<float> h_output(kElementCount);

  float* d_input = nullptr;
  float* d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, kElementCount * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, kElementCount * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), kElementCount * sizeof(float),
                        cudaMemcpyHostToDevice));

  brent_kung_scan<<<1, kBlockSize>>>(d_output, d_input, kElementCount);
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
