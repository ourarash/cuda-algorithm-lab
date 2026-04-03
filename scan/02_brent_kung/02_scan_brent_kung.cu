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
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

constexpr int kBlockSize = 1024;
constexpr int kElementCount = 1024;

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                        \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

void cpu_inclusive_scan(const float* input, float* output, int n) {
  output[0] = input[0];
  for (int i = 1; i < n; ++i) {
    output[i] = output[i - 1] + input[i];
  }
}

bool almost_equal(const float* a, const float* b, int n, float eps = 1e-4f) {
  for (int i = 0; i < n; ++i) {
    if (std::fabs(a[i] - b[i]) > eps) {
      std::cerr << "Mismatch at index " << i << ": GPU=" << b[i]
                << ", CPU=" << a[i] << '\n';
      return false;
    }
  }
  return true;
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

  cudaDeviceProp prop;
  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
  std::cout << "Running on GPU: " << prop.name << " (Compute Capability "
            << prop.major << "." << prop.minor << ")\n";

  std::vector<float> h_input(kElementCount);
  std::vector<float> h_output(kElementCount);
  std::vector<float> h_reference(kElementCount);

  std::mt19937 gen(19);
  std::uniform_real_distribution<float> dist(1.0f, 1.1f);
  for (float& value : h_input) {
    value = dist(gen);
  }

  float* d_input = nullptr;
  float* d_output = nullptr;
  CHECK_CUDA(cudaMalloc(&d_input, kElementCount * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_output, kElementCount * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_input, h_input.data(),
                        kElementCount * sizeof(float),
                        cudaMemcpyHostToDevice));

  brent_kung_scan<<<1, kBlockSize>>>(d_output, d_input, kElementCount);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(h_output.data(), d_output,
                        kElementCount * sizeof(float),
                        cudaMemcpyDeviceToHost));

  cpu_inclusive_scan(h_input.data(), h_reference.data(), kElementCount);

  std::cout << "Last GPU output: " << h_output.back() << '\n';
  std::cout << "Last CPU output: " << h_reference.back() << '\n';
  std::cout << (almost_equal(h_reference.data(), h_output.data(), kElementCount)
                    ? "CPU and GPU results match.\n"
                    : "CPU and GPU results differ.\n");

  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_output));
  return 0;
}
