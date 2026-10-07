/*
 * Shared-Memory Matrix Transpose
 *
 * Intention:
 * This file shows a simple tiled matrix transpose in CUDA.
 *
 * High-Level Algorithm:
 * - Launch one 32 x 32 thread block per matrix tile.
 * - Load a tile from global memory into shared memory with coalesced reads.
 * - Synchronize the block, then read the shared tile with swapped indices.
 * - Write the transposed tile back to global memory with coalesced writes.
 * - Pad the shared tile to avoid bank conflicts during the transposed read.
 */
#include <cstdio>
#include <vector>

#include "lab.cuh"

constexpr int TILE_SIZE = 32;

/**
 * 0. Shared-Memory Transpose with Padding
 * The extra shared-memory column changes the row stride from 32 to 33 floats,
 * which avoids bank conflicts during the transposed shared-memory read.
 */
__global__ void transpose_shared_memory(const float *input, float *output,
                                        int rows, int cols) {
  __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

  const int input_col = blockIdx.x * blockDim.x + threadIdx.x;
  const int input_row = blockIdx.y * blockDim.y + threadIdx.y;

  if (input_row < rows && input_col < cols) {
    tile[threadIdx.y][threadIdx.x] = input[input_row * cols + input_col];
  } else {
    tile[threadIdx.y][threadIdx.x] = 0.0f;
  }

  __syncthreads();

  const int output_col = blockIdx.y * blockDim.x + threadIdx.x;
  const int output_row = blockIdx.x * blockDim.y + threadIdx.y;

  if (output_row < cols && output_col < rows) {
    output[output_row * rows + output_col] = tile[threadIdx.x][threadIdx.y];
  }
}

void cpu_transpose(int rows, int cols, const float *input, float *output) {
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      output[col * rows + row] = input[row * cols + col];
    }
  }
}

int main(int argc, char **argv) {
  lab::Args args(argc, argv);
  // The quick size is neither square nor a multiple of the tile size, which
  // exercises the bounds checks on both the read and the write side.
  const int rows = static_cast<int>(args.get_int("rows", args.quick() ? 1000 : 4096));
  const int cols = static_cast<int>(args.get_int("cols", args.quick() ? 777 : 4096));
  const size_t count = static_cast<size_t>(rows) * cols;

  lab::print_device();
  printf("Shared-memory matrix transpose (%d x %d)\n", rows, cols);

  const std::vector<float> input = lab::random_uniform<float>(count, -1.f, 1.f, 3);
  std::vector<float> output_cpu(count);
  std::vector<float> output_gpu(count);
  cpu_transpose(rows, cols, input.data(), output_cpu.data());

  float *d_input;
  float *d_output;
  CUDA_CHECK(cudaMalloc(&d_input, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), count * sizeof(float),
                        cudaMemcpyHostToDevice));

  dim3 block_dim(TILE_SIZE, TILE_SIZE);
  dim3 grid_dim(lab::ceil_div(cols, TILE_SIZE), lab::ceil_div(rows, TILE_SIZE));

  transpose_shared_memory<<<grid_dim, block_dim>>>(d_input, d_output, rows,
                                                   cols);
  CUDA_CHECK_LAUNCH();
  CUDA_CHECK(cudaMemcpy(output_gpu.data(), d_output, count * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // A transpose only moves values, so the result must match exactly.
  const bool pass = lab::check_equal("transposed matrix", output_gpu, output_cpu);

  // A transpose reads and writes every element once; a plain copy of the
  // same size is the speed limit it is measured against.
  const float ms = lab::time_ms([&] {
    transpose_shared_memory<<<grid_dim, block_dim>>>(d_input, d_output, rows,
                                                     cols);
  });
  lab::report("shared-memory transpose", ms, 0, 2.0 * count * sizeof(float));
  const float copy_ms = lab::time_ms([&] {
    CUDA_CHECK(cudaMemcpy(d_output, d_input, count * sizeof(float),
                          cudaMemcpyDeviceToDevice));
  });
  lab::report("cudaMemcpy (device to device)", copy_ms, 0,
              2.0 * count * sizeof(float));

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  return lab::finish(pass);
}
