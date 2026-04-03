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

void cpu_exclusive_scan(const float* input, float* output, int n) {
  output[0] = 0.0f;
  for (int i = 1; i < n; ++i) {
    output[i] = output[i - 1] + input[i - 1];
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

  cudaDeviceProp prop;
  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
  std::cout << "Running on GPU: " << prop.name << " (Compute Capability "
            << prop.major << "." << prop.minor << ")\n";

  std::vector<float> h_input(kElementCount);
  std::vector<float> h_output(kElementCount);
  std::vector<float> h_reference(kElementCount);

  std::mt19937 gen(23);
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

  blelloch_scan<<<1, kBlockSize>>>(d_output, d_input, kElementCount);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(h_output.data(), d_output,
                        kElementCount * sizeof(float),
                        cudaMemcpyDeviceToHost));

  cpu_exclusive_scan(h_input.data(), h_reference.data(), kElementCount);

  std::cout << "Last GPU output: " << h_output.back() << '\n';
  std::cout << "Last CPU output: " << h_reference.back() << '\n';
  std::cout << (almost_equal(h_reference.data(), h_output.data(), kElementCount)
                    ? "CPU and GPU results match.\n"
                    : "CPU and GPU results differ.\n");

  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_output));
  return 0;
}
