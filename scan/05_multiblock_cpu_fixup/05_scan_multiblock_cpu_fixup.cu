/*
 * Multi-Block Inclusive Scan with CPU Fixup
 *
 * High-Level Algorithm:
 * A single CUDA block can only scan a limited number of elements. To handle a
 * large array, we break the work into block-sized chunks.
 *
 * Phase 1 (Per-Block GPU Scan):
 * - Each block performs a shared-memory inclusive scan over its own chunk.
 * - The last value in each block is written out as that block's total sum.
 *
 * Phase 2 (CPU Scan of Block Totals):
 * - The array of block totals is copied to the host.
 * - The CPU scans those block totals to compute each block's carry-in offset.
 *
 * Phase 3 (Host Fixup):
 * - The scanned offset from the previous block is added to every element in the
 *   current block.
 *
 * Why keep this version:
 * - It is the easiest large-array extension to understand.
 * - It makes the transition from single-block scan to fully recursive GPU scan
 *   very explicit.
 */
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

constexpr int kBlockSize = 1024;
constexpr int kElementCount = 1 << 20;

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                        \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

template <typename T>
constexpr T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

void cpu_inclusive_scan(const double* input, double* output, int n) {
  output[0] = input[0];
  for (int i = 1; i < n; ++i) {
    output[i] = output[i - 1] + input[i];
  }
}

bool almost_equal(const double* a, const double* b, int n, double eps = 1e-8) {
  for (int i = 0; i < n; ++i) {
    double scale = std::max({1.0, std::fabs(a[i]), std::fabs(b[i])});
    if (std::fabs(a[i] - b[i]) > eps * scale) {
      std::cerr << "Mismatch at index " << i << ": GPU=" << b[i]
                << ", CPU=" << a[i] << '\n';
      return false;
    }
  }
  return true;
}

__global__ void block_scan(double* output, double* block_sums,
                           const double* input, int n) {
  __shared__ double shared[2][kBlockSize];

  const int tid = threadIdx.x;
  const int global_index = blockIdx.x * blockDim.x + tid;

  shared[0][tid] = (global_index < n) ? input[global_index] : 0.0;
  __syncthreads();

  int current = 0;
  for (int offset = 1; offset < blockDim.x; offset *= 2) {
    const int previous = current;
    current = 1 - current;

    double value = shared[previous][tid];
    if (tid >= offset) {
      value += shared[previous][tid - offset];
    }
    shared[current][tid] = value;
    __syncthreads();
  }

  if (global_index < n) {
    output[global_index] = shared[current][tid];
  }

  if (tid == blockDim.x - 1) {
    block_sums[blockIdx.x] = shared[current][tid];
  }
}

int main() {
  static_assert(kBlockSize <= 1024, "CUDA thread blocks cannot exceed 1024.");

  const int num_blocks = ceil_div(kElementCount, kBlockSize);

  cudaDeviceProp prop;
  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
  std::cout << "Running on GPU: " << prop.name << " (Compute Capability "
            << prop.major << "." << prop.minor << ")\n";

  std::vector<double> h_input(kElementCount);
  std::vector<double> h_output(kElementCount);
  std::vector<double> h_reference(kElementCount);
  std::vector<double> h_block_sums(num_blocks);
  std::vector<double> h_scanned_block_sums(num_blocks);

  std::mt19937 gen(31);
  std::uniform_real_distribution<double> dist(1.0, 1.1);
  for (double& value : h_input) {
    value = dist(gen);
  }

  double* d_input = nullptr;
  double* d_output = nullptr;
  double* d_block_sums = nullptr;
  CHECK_CUDA(cudaMalloc(&d_input, kElementCount * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&d_output, kElementCount * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&d_block_sums, num_blocks * sizeof(double)));
  CHECK_CUDA(cudaMemcpy(d_input, h_input.data(),
                        kElementCount * sizeof(double),
                        cudaMemcpyHostToDevice));

  block_scan<<<num_blocks, kBlockSize>>>(d_output, d_block_sums, d_input,
                                         kElementCount);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(h_output.data(), d_output,
                        kElementCount * sizeof(double),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_block_sums.data(), d_block_sums,
                        num_blocks * sizeof(double),
                        cudaMemcpyDeviceToHost));

  cpu_inclusive_scan(h_block_sums.data(), h_scanned_block_sums.data(),
                     num_blocks);

  for (int block = 1; block < num_blocks; ++block) {
    double carry_in = h_scanned_block_sums[block - 1];
    int block_begin = block * kBlockSize;
    int block_end = std::min(block_begin + kBlockSize, kElementCount);
    for (int i = block_begin; i < block_end; ++i) {
      h_output[i] += carry_in;
    }
  }

  cpu_inclusive_scan(h_input.data(), h_reference.data(), kElementCount);

  std::cout << "Last GPU output: " << h_output.back() << '\n';
  std::cout << "Last CPU output: " << h_reference.back() << '\n';
  std::cout << (almost_equal(h_reference.data(), h_output.data(), kElementCount)
                    ? "CPU and GPU results match.\n"
                    : "CPU and GPU results differ.\n");

  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_output));
  CHECK_CUDA(cudaFree(d_block_sums));
  return 0;
}
