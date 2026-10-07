/*
 * COO Sparse Matrix-Vector Multiply
 *
 * Intention:
 * This file demonstrates the simplest sparse matrix-vector multiply format:
 * coordinate (COO) storage, where every non-zero stores its row, column, and
 * value explicitly.
 *
 * High-Level Algorithm:
 * - Convert a dense test matrix into COO triples.
 * - Launch one thread per non-zero entry.
 * - Each thread multiplies its non-zero by the matching input-vector entry.
 * - Atomically accumulate the contribution into the output row.
 * - Compare the GPU result against a dense CPU reference.
 */
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

#include "lab.cuh"

constexpr int kBlockSize = 256;

struct CooMatrix {
  int rows = 0;
  int cols = 0;
  std::vector<int> row_indices;
  std::vector<int> col_indices;
  std::vector<float> values;
};

CooMatrix dense_to_coo(const std::vector<std::vector<float>>& dense) {
  CooMatrix matrix;
  matrix.rows = static_cast<int>(dense.size());
  matrix.cols = matrix.rows == 0 ? 0 : static_cast<int>(dense[0].size());

  for (int row = 0; row < matrix.rows; ++row) {
    for (int col = 0; col < matrix.cols; ++col) {
      if (dense[row][col] != 0.0f) {
        matrix.row_indices.push_back(row);
        matrix.col_indices.push_back(col);
        matrix.values.push_back(dense[row][col]);
      }
    }
  }

  return matrix;
}

std::vector<std::vector<float>> initialize_dense_matrix(int rows, int cols,
                                                        float sparsity) {
  std::vector<std::vector<float>> matrix(rows, std::vector<float>(cols, 0.0f));
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> value_dist(0.0f, 1.0f);
  std::uniform_real_distribution<float> keep_dist(0.0f, 1.0f);

  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      if (keep_dist(rng) > sparsity) {
        matrix[row][col] = value_dist(rng);
      }
    }
  }

  return matrix;
}

std::vector<float> initialize_vector(int size, float value) {
  return std::vector<float>(size, value);
}

// Dense CPU reference in double precision.
std::vector<double> spmv_reference(const std::vector<std::vector<float>>& matrix,
                                   const std::vector<float>& input) {
  std::vector<double> output(matrix.size(), 0.0);
  for (size_t row = 0; row < matrix.size(); ++row) {
    for (size_t col = 0; col < matrix[row].size(); ++col) {
      output[row] += static_cast<double>(matrix[row][col]) * input[col];
    }
  }
  return output;
}

__global__ void spmv_coo_kernel(int nnz, const int* row_indices,
                                const int* col_indices, const float* values,
                                const float* input, float* output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nnz) {
    int row = row_indices[idx];
    int col = col_indices[idx];
    atomicAdd(&output[row], values[idx] * input[col]);
  }
}

int main() {
  lab::print_device();
  const int rows = 512;
  const int cols = 512;
  const float sparsity = 0.98f;

  std::vector<std::vector<float>> dense = initialize_dense_matrix(rows, cols, sparsity);
  CooMatrix coo = dense_to_coo(dense);
  std::vector<float> h_input = initialize_vector(cols, 1.25f);
  std::vector<float> h_output(rows, 0.0f);
  std::vector<double> h_reference = spmv_reference(dense, h_input);

  int* d_row_indices = nullptr;
  int* d_col_indices = nullptr;
  float* d_values = nullptr;
  float* d_input = nullptr;
  float* d_output = nullptr;

  CUDA_CHECK(cudaMalloc(&d_row_indices, coo.row_indices.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_col_indices, coo.col_indices.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_values, coo.values.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, h_output.size() * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_row_indices, coo.row_indices.data(),
                        coo.row_indices.size() * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_col_indices, coo.col_indices.data(),
                        coo.col_indices.size() * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, coo.values.data(),
                        coo.values.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_output, 0, h_output.size() * sizeof(float)));

  int nnz = static_cast<int>(coo.values.size());
  int grid_size = (nnz + kBlockSize - 1) / kBlockSize;
  spmv_coo_kernel<<<grid_size, kBlockSize>>>(nnz, d_row_indices, d_col_indices,
                                             d_values, d_input, d_output);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_output.data(), d_output,
                        h_output.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  std::cout << "Non-zeros: " << nnz << '\n';
  // Atomic additions arrive in an unpredictable order, so the float sums can
  // differ from the reference in the last bits; nothing more.
  const bool pass = lab::check_close("y = A x", h_output, h_reference, 1e-5, 1e-5);

  CUDA_CHECK(cudaFree(d_row_indices));
  CUDA_CHECK(cudaFree(d_col_indices));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  return lab::finish(pass);
}
