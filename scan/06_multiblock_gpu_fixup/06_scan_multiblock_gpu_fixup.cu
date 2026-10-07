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
 * Why this version is the end state of this folder:
 * - It keeps the entire computation on the GPU.
 * - It works for arrays much larger than a single block. The default size
 *   needs two levels of recursion.
 *
 * Every recursion level needs scratch arrays for its block sums. They are
 * allocated once, before any kernel runs: cudaMalloc and cudaFree are slow
 * and wait for the whole device to go idle, so calling them inside the scan
 * would dominate its run time.
 *
 * The data is double precision so that, over millions of elements, rounding
 * differences between the GPU and CPU summation orders stay negligible.
 */
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

__global__ void add_block_offsets(double* output, const double* scanned_sums,
                                  int n) {
  const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_index >= n || blockIdx.x == 0) {
    return;
  }

  output[global_index] += scanned_sums[blockIdx.x - 1];
}

// Scratch space for every recursion level: level l holds the block sums of
// the level-l problem and their scan, which is the level-(l+1) output.
struct ScanWorkspace {
  std::vector<double*> block_sums;
  std::vector<double*> scanned_block_sums;
  double* unused_block_sum = nullptr;  // For the final single-block level.
};

ScanWorkspace allocate_workspace(int n) {
  ScanWorkspace ws;
  for (int count = lab::ceil_div(n, kBlockSize); count > 1;
       count = lab::ceil_div(count, kBlockSize)) {
    double* sums = nullptr;
    double* scanned = nullptr;
    CUDA_CHECK(cudaMalloc(&sums, count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&scanned, count * sizeof(double)));
    ws.block_sums.push_back(sums);
    ws.scanned_block_sums.push_back(scanned);
  }
  CUDA_CHECK(cudaMalloc(&ws.unused_block_sum, sizeof(double)));
  return ws;
}

void free_workspace(ScanWorkspace& ws) {
  for (size_t level = 0; level < ws.block_sums.size(); ++level) {
    CUDA_CHECK(cudaFree(ws.block_sums[level]));
    CUDA_CHECK(cudaFree(ws.scanned_block_sums[level]));
  }
  CUDA_CHECK(cudaFree(ws.unused_block_sum));
}

void inclusive_scan_gpu(double* d_output, const double* d_input, int n,
                        const ScanWorkspace& ws, int level = 0) {
  const int num_blocks = lab::ceil_div(n, kBlockSize);

  if (num_blocks == 1) {
    block_scan<<<1, kBlockSize>>>(d_output, ws.unused_block_sum, d_input, n);
    CUDA_CHECK_LAUNCH();
    return;
  }

  double* d_block_sums = ws.block_sums[level];
  double* d_scanned_block_sums = ws.scanned_block_sums[level];

  block_scan<<<num_blocks, kBlockSize>>>(d_output, d_block_sums, d_input, n);
  CUDA_CHECK_LAUNCH();

  inclusive_scan_gpu(d_scanned_block_sums, d_block_sums, num_blocks, ws,
                     level + 1);

  add_block_offsets<<<num_blocks, kBlockSize>>>(d_output, d_scanned_block_sums,
                                                n);
  CUDA_CHECK_LAUNCH();
}

int main(int argc, char** argv) {
  static_assert(kBlockSize <= 1024, "CUDA thread blocks cannot exceed 1024.");
  lab::Args args(argc, argv);
  const int n =
      static_cast<int>(args.get_int("n", args.quick() ? 100003 : (1 << 21) + 12345));

  lab::print_device();
  ScanWorkspace ws = allocate_workspace(n);
  printf("Inclusive scan of %d doubles (%zu recursion levels)\n", n,
         ws.block_sums.size() + 1);

  const std::vector<double> h_input = lab::random_uniform<double>(n, 1.0, 1.1, 37);
  std::vector<double> h_output(n);

  double* d_input = nullptr;
  double* d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), n * sizeof(double),
                        cudaMemcpyHostToDevice));

  inclusive_scan_gpu(d_output, d_input, n, ws);
  CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, n * sizeof(double),
                        cudaMemcpyDeviceToHost));

  const std::vector<double> expected = cpu_inclusive_scan(h_input);
  printf("Last GPU output: %.6f\n", h_output.back());
  printf("Last CPU output: %.6f\n", expected.back());
  const bool pass = lab::check_close("scan", h_output, expected, 1e-9, 1e-9);

  // Effective bandwidth: the minimum traffic is one read and one write of
  // the array. This implementation reads and writes the output a second time
  // in the offset fixup, which is what single-pass scans eliminate.
  const float ms =
      lab::time_ms([&] { inclusive_scan_gpu(d_output, d_input, n, ws); });
  lab::report("recursive multi-block scan", ms, 0, 2.0 * n * sizeof(double));

  free_workspace(ws);
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  return lab::finish(pass);
}
