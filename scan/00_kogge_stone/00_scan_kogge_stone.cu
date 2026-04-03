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

  cudaDeviceProp prop;
  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
  std::cout << "Running on GPU: " << prop.name << " (Compute Capability "
            << prop.major << "." << prop.minor << ")\n";

  std::vector<float> h_input(kElementCount);
  std::vector<float> h_output(kElementCount);
  std::vector<float> h_reference(kElementCount);

  std::mt19937 gen(7);
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

  kogge_stone_scan<<<1, kBlockSize>>>(d_output, d_input, kElementCount);
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
