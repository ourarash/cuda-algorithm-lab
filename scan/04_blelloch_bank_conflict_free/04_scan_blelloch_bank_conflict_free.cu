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
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

constexpr int kBlockSize = 1024;
constexpr int kElementCount = 1024;
constexpr int kNumBanks = 32;
constexpr int kPadding = kBlockSize / kNumBanks;
constexpr int kPaddedSize = kBlockSize + kPadding;

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                        \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

__host__ __device__ constexpr int conflict_free_index(int index) {
  return index + (index / kNumBanks);
}

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

  cudaDeviceProp prop;
  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
  std::cout << "Running on GPU: " << prop.name << " (Compute Capability "
            << prop.major << "." << prop.minor << ")\n";

  std::vector<float> h_input(kElementCount);
  std::vector<float> h_output(kElementCount);
  std::vector<float> h_reference(kElementCount);

  std::mt19937 gen(29);
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

  blelloch_scan_padded<<<1, kBlockSize>>>(d_output, d_input, kElementCount);
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
