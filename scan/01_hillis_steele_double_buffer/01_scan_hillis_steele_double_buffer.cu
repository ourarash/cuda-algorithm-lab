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

  cudaDeviceProp prop;
  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
  std::cout << "Running on GPU: " << prop.name << " (Compute Capability "
            << prop.major << "." << prop.minor << ")\n";

  std::vector<float> h_input(kElementCount);
  std::vector<float> h_output(kElementCount);
  std::vector<float> h_reference(kElementCount);

  std::mt19937 gen(11);
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

  hillis_steele_scan<<<1, kBlockSize>>>(d_output, d_input, kElementCount);
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
