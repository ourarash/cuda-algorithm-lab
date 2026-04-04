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
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "     \
                << cudaGetErrorString(err) << std::endl;                       \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))

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

bool nearly_equal(float a, float b, float eps = 1e-5f) {
  return std::fabs(a - b) < eps;
}

bool validate_transpose(const std::vector<float> &reference,
                        const std::vector<float> &candidate) {
  for (size_t i = 0; i < reference.size(); ++i) {
    if (!nearly_equal(reference[i], candidate[i])) {
      std::cerr << "Mismatch at " << i << ": CPU=" << reference[i]
                << ", GPU=" << candidate[i] << std::endl;
      return false;
    }
  }
  return true;
}

int main() {
  const int rows = 4096;
  const int cols = 4096;

  std::vector<float> input(rows * cols);
  std::vector<float> output_cpu(cols * rows);
  std::vector<float> output_gpu(cols * rows);

  for (int i = 0; i < rows * cols; ++i) {
    input[i] = static_cast<float>(i % 1000) * 0.001f;
  }

  std::cout << "Running CPU validation..." << std::endl;
  cpu_transpose(rows, cols, input.data(), output_cpu.data());

  float *d_input;
  float *d_output;
  CHECK(cudaMalloc(&d_input, input.size() * sizeof(float)));
  CHECK(cudaMalloc(&d_output, output_gpu.size() * sizeof(float)));

  CHECK(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float),
                   cudaMemcpyHostToDevice));

  dim3 block_dim(TILE_SIZE, TILE_SIZE);
  dim3 grid_dim(CEIL_DIV(cols, TILE_SIZE), CEIL_DIV(rows, TILE_SIZE));

  transpose_shared_memory<<<grid_dim, block_dim>>>(d_input, d_output, rows,
                                                   cols);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaMemcpy(output_gpu.data(), d_output,
                   output_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
  const bool pass = validate_transpose(output_cpu, output_gpu);

  std::cout << "Shared-Memory Matrix Transpose (" << rows << " x " << cols
            << ")\n";
  std::cout << "Validation " << (pass ? "PASSED" : "FAILED") << std::endl;

  CHECK(cudaFree(d_input));
  CHECK(cudaFree(d_output));
  return pass ? 0 : 1;
}
