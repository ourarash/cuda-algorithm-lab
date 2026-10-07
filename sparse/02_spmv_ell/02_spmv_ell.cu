/*
 * ELL (ELLPACK) Sparse Matrix-Vector Multiply
 *
 * Intention:
 * This file shows how padding a sparse matrix into a regular shape fixes the
 * uncoalesced loads of the CSR scalar kernel.
 *
 * The ELL format:
 * - Let K be the largest number of non-zeros in any row.
 * - Every row is padded to exactly K entries. Padding entries have value 0
 *   (and column 0), so they add nothing to the result.
 * - The rows x K arrays of values and column indices are stored
 *   column-major: entry `slot` of row `row` lives at index slot * rows + row.
 *
 * High-Level Algorithm:
 * - Launch one thread per row, as in CSR scalar.
 * - In loop iteration `slot`, thread `row` reads index slot * rows + row.
 *   Consecutive threads read consecutive addresses, so the loads of values
 *   and column indices are coalesced.
 * - Every thread runs exactly K iterations, so warps do not diverge.
 *
 * Trade-off:
 * Padding costs memory and bandwidth. If one row is much longer than the rest
 * (common in real matrices such as web graphs), almost all of the stored
 * entries are padding. The program prints the fraction of useful entries.
 * Hybrid ELL + COO formats exist for exactly that case.
 */
#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

#include "lab.cuh"

#define BLOCK_SIZE 256

struct EllMatrix {
  int rows = 0;
  int cols = 0;
  int entries_per_row = 0;   // K, the longest row
  std::vector<int> colInd;   // rows * K, column-major
  std::vector<float> values; // rows * K, column-major
};

EllMatrix dense_to_ell(const std::vector<std::vector<float>>& dense) {
  EllMatrix ell;
  ell.rows = static_cast<int>(dense.size());
  ell.cols = ell.rows == 0 ? 0 : static_cast<int>(dense[0].size());

  // Gather each row's non-zeros, and find the longest row.
  std::vector<std::vector<int>> row_cols(ell.rows);
  for (int r = 0; r < ell.rows; ++r) {
    for (int c = 0; c < ell.cols; ++c) {
      if (dense[r][c] != 0.0f) {
        row_cols[r].push_back(c);
      }
    }
    ell.entries_per_row =
        std::max(ell.entries_per_row, static_cast<int>(row_cols[r].size()));
  }

  // Lay the padded rows out column-major. Padding slots keep the value 0 and
  // column 0, so the kernel can process them without a branch.
  const size_t total = static_cast<size_t>(ell.rows) * ell.entries_per_row;
  ell.colInd.assign(total, 0);
  ell.values.assign(total, 0.0f);
  for (int r = 0; r < ell.rows; ++r) {
    for (size_t slot = 0; slot < row_cols[r].size(); ++slot) {
      const size_t idx = slot * ell.rows + r;
      ell.colInd[idx] = row_cols[r][slot];
      ell.values[idx] = dense[r][row_cols[r][slot]];
    }
  }
  return ell;
}

std::vector<std::vector<float>> initializeDenseMatrix(int m, int n,
                                                      float sparsity) {
  std::vector<std::vector<float>> matrix(m, std::vector<float>(n, 0.0f));
  std::mt19937 rng(42);  // Fixed seed for reproducibility
  std::uniform_real_distribution<float> dist_val(0.0f, 1.0f);
  std::uniform_real_distribution<float> dist_prob(0.0f, 1.0f);

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      if (dist_prob(rng) > sparsity) {
        matrix[i][j] = dist_val(rng);
      }
    }
  }
  return matrix;
}

// One thread per row; every row has exactly `entries_per_row` slots.
__global__ void spmv_ell_kernel(int rows, int entries_per_row,
                                const int* colInd, const float* values,
                                const float* x, float* y) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows) {
    float sum = 0.0f;
    for (int slot = 0; slot < entries_per_row; ++slot) {
      const int idx = slot * rows + row;  // Column-major: coalesced
      sum += values[idx] * x[colInd[idx]];
    }
    y[row] = sum;
  }
}

// Dense CPU reference in double precision.
std::vector<double> spmv_reference(const std::vector<std::vector<float>>& A,
                                   const std::vector<float>& x) {
  std::vector<double> y(A.size(), 0.0);
  for (size_t i = 0; i < A.size(); ++i) {
    for (size_t j = 0; j < A[i].size(); ++j) {
      y[i] += static_cast<double>(A[i][j]) * x[j];
    }
  }
  return y;
}

int main() {
  lab::print_device();
  const int rows = 1000;
  const int cols = 1000;
  const auto dense = initializeDenseMatrix(rows, cols, /*sparsity=*/0.99f);
  const EllMatrix ell = dense_to_ell(dense);

  size_t nnz = 0;
  for (float v : ell.values) {
    nnz += (v != 0.0f);
  }
  const size_t stored = ell.values.size();
  printf("ELL SpMV: %d x %d matrix, %zu non-zeros, K = %d entries per row\n",
         rows, cols, nnz, ell.entries_per_row);
  printf("Stored entries: %zu (%.0f%% useful, the rest is padding)\n", stored,
         stored ? 100.0 * nnz / stored : 100.0);

  const std::vector<float> h_x(cols, 1.12f);
  std::vector<float> h_y(rows);

  float *d_x, *d_y, *d_values;
  int* d_colInd;
  CUDA_CHECK(cudaMalloc(&d_x, cols * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, rows * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_colInd, std::max<size_t>(stored, 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_values, std::max<size_t>(stored, 1) * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), cols * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_colInd, ell.colInd.data(), stored * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, ell.values.data(), stored * sizeof(float),
                        cudaMemcpyHostToDevice));

  spmv_ell_kernel<<<lab::ceil_div(rows, BLOCK_SIZE), BLOCK_SIZE>>>(
      rows, ell.entries_per_row, d_colInd, d_values, d_x, d_y);
  CUDA_CHECK_LAUNCH();
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, rows * sizeof(float),
                        cudaMemcpyDeviceToHost));

  printf("First few results: %f %f %f %f %f\n", h_y[0], h_y[1], h_y[2], h_y[3],
         h_y[4]);
  const bool pass = lab::check_close("y = A x", h_y, spmv_reference(dense, h_x),
                                     1e-5, 1e-5);

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaFree(d_colInd));
  CUDA_CHECK(cudaFree(d_values));
  return lab::finish(pass);
}
