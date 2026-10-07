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
 *
 * The data is double precision so that, over a million elements, rounding
 * differences between the GPU and CPU summation orders stay negligible.
 */
#include <algorithm>
#include <cstdio>
#include <vector>

#include "lab.cuh"

constexpr int kBlockSize = 1024;

std::vector<double> cpu_inclusive_scan(const std::vector<double>& input) {
  std::vector<double> output(input.size());
  double running = 0.0;
  for (size_t i = 0; i < input.size(); ++i) {
    running += input[i];
    output[i] = running;
  }
  return output;
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

int main(int argc, char** argv) {
  lab::Args args(argc, argv);
  const int n =
      static_cast<int>(args.get_int("n", args.quick() ? 100003 : (1 << 20) + 12345));
  const int num_blocks = lab::ceil_div(n, kBlockSize);

  lab::print_device();
  printf("Inclusive scan of %d doubles in %d blocks\n", n, num_blocks);

  const std::vector<double> h_input = lab::random_uniform<double>(n, 1.0, 1.1, 31);
  std::vector<double> h_output(n);
  std::vector<double> h_block_sums(num_blocks);

  double* d_input = nullptr;
  double* d_output = nullptr;
  double* d_block_sums = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_block_sums, num_blocks * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), n * sizeof(double),
                        cudaMemcpyHostToDevice));

  // Phase 1 on the GPU.
  block_scan<<<num_blocks, kBlockSize>>>(d_output, d_block_sums, d_input, n);
  CUDA_CHECK_LAUNCH();

  CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, n * sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_block_sums.data(), d_block_sums,
                        num_blocks * sizeof(double), cudaMemcpyDeviceToHost));

  // Phases 2 and 3 on the CPU.
  const std::vector<double> scanned_block_sums = cpu_inclusive_scan(h_block_sums);
  for (int block = 1; block < num_blocks; ++block) {
    const double carry_in = scanned_block_sums[block - 1];
    const int block_begin = block * kBlockSize;
    const int block_end = std::min(block_begin + kBlockSize, n);
    for (int i = block_begin; i < block_end; ++i) {
      h_output[i] += carry_in;
    }
  }

  const std::vector<double> expected = cpu_inclusive_scan(h_input);
  printf("Last GPU output: %.6f\n", h_output.back());
  printf("Last CPU output: %.6f\n", expected.back());
  const bool pass = lab::check_close("scan", h_output, expected, 1e-9, 1e-9);

  // Only the GPU phase is timed; the CPU fixup and the copies it needs are the
  // reason the next version moves everything onto the GPU.
  const float ms = lab::time_ms([&] {
    block_scan<<<num_blocks, kBlockSize>>>(d_output, d_block_sums, d_input, n);
  });
  lab::report("per-block scan (GPU phase)", ms, 0,
              2.0 * n * sizeof(double));

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_block_sums));
  return lab::finish(pass);
}
