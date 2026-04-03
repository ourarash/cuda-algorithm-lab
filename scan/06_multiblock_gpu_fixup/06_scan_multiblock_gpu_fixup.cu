/*
 * Multi-Block Inclusive Scan with Recursive GPU Fixup
 *
 * High-Level Algorithm:
 * This is the fully GPU-resident version of the large-array scan.
 *
 * Phase 1 (Per-Block Scan):
 * - Each block scans its own chunk in shared memory.
 * - The last element of each block becomes that block's total sum.
 *
 * Phase 2 (Recursive Scan of Block Sums):
 * - The block sums themselves form a smaller scan problem.
 * - We solve that smaller problem with the same routine recursively until only
 *   one block remains.
 *
 * Phase 3 (Add Block Offsets):
 * - Once the block sums are scanned, block i adds the scanned total of block
 *   i - 1 to every element in its local output.
 *
 * Why this version is the end state:
 * - It keeps the entire computation on the GPU.
 * - It works for arrays much larger than a single block.
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

__global__ void add_block_offsets(double* output, const double* scanned_sums,
                                  int n) {
  const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_index >= n || blockIdx.x == 0) {
    return;
  }

  output[global_index] += scanned_sums[blockIdx.x - 1];
}

void inclusive_scan_gpu(double* d_output, const double* d_input, int n) {
  const int num_blocks = ceil_div(n, kBlockSize);

  if (num_blocks == 1) {
    double* d_single_block_sum = nullptr;
    CHECK_CUDA(cudaMalloc(&d_single_block_sum, sizeof(double)));
    block_scan<<<1, kBlockSize>>>(d_output, d_single_block_sum, d_input, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaFree(d_single_block_sum));
    return;
  }

  double* d_block_sums = nullptr;
  double* d_scanned_block_sums = nullptr;
  CHECK_CUDA(cudaMalloc(&d_block_sums, num_blocks * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&d_scanned_block_sums, num_blocks * sizeof(double)));

  block_scan<<<num_blocks, kBlockSize>>>(d_output, d_block_sums, d_input, n);
  CHECK_CUDA(cudaGetLastError());

  inclusive_scan_gpu(d_scanned_block_sums, d_block_sums, num_blocks);

  add_block_offsets<<<num_blocks, kBlockSize>>>(d_output, d_scanned_block_sums,
                                                n);
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaFree(d_block_sums));
  CHECK_CUDA(cudaFree(d_scanned_block_sums));
}

int main() {
  static_assert(kBlockSize <= 1024, "CUDA thread blocks cannot exceed 1024.");

  cudaDeviceProp prop;
  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
  std::cout << "Running on GPU: " << prop.name << " (Compute Capability "
            << prop.major << "." << prop.minor << ")\n";

  std::vector<double> h_input(kElementCount);
  std::vector<double> h_output(kElementCount);
  std::vector<double> h_reference(kElementCount);

  std::mt19937 gen(37);
  std::uniform_real_distribution<double> dist(1.0, 1.1);
  for (double& value : h_input) {
    value = dist(gen);
  }

  double* d_input = nullptr;
  double* d_output = nullptr;
  CHECK_CUDA(cudaMalloc(&d_input, kElementCount * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&d_output, kElementCount * sizeof(double)));
  CHECK_CUDA(cudaMemcpy(d_input, h_input.data(),
                        kElementCount * sizeof(double),
                        cudaMemcpyHostToDevice));

  inclusive_scan_gpu(d_output, d_input, kElementCount);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(h_output.data(), d_output,
                        kElementCount * sizeof(double),
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
