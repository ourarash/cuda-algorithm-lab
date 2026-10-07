/*
 * CSR Sparse Matrix-Vector Multiply
 *
 * Intention:
 * This file demonstrates compressed sparse row (CSR) storage, the most widely
 * used general-purpose sparse format, and the simplest CSR SpMV kernel.
 *
 * CSR stores three arrays:
 * - values[nnz]:   the non-zero values, row by row
 * - colInd[nnz]:   the column of each value
 * - rowPtr[rows+1]: where each row starts in values/colInd; row r occupies
 *                   [rowPtr[r], rowPtr[r + 1])
 * Compared with COO, the per-element row index is replaced by one offset per
 * row, and every row's elements are contiguous.
 *
 * High-Level Algorithm ("CSR scalar"):
 * - Launch one thread per row.
 * - Each thread walks its row's non-zeros and accumulates a dot product with
 *   the input vector in a register, then writes y[row] once.
 * - No atomics are needed, because each output element has exactly one owner.
 *
 * Limitation:
 * Adjacent threads walk different rows, so their reads of values/colInd are
 * not coalesced, and a few long rows can keep one thread busy while the rest
 * of its warp idles. Assigning a warp per row ("CSR vector") and the ELL
 * format (next example) address these.
 */
#include <cstdio>
#include <random>
#include <vector>

#include "lab.cuh"

#define BLOCK_SIZE 256

template <typename T>
class CSRMatrix {
 public:
  std::vector<int> rowPtr;
  std::vector<int> colInd;
  std::vector<T> values;

  explicit CSRMatrix(const std::vector<std::vector<T>>& denseMatrix) {
    denseToCSR(denseMatrix);
  }

 private:
  void denseToCSR(const std::vector<std::vector<T>>& denseMatrix) {
    rowPtr.clear();
    colInd.clear();
    values.clear();

    const int numRows = static_cast<int>(denseMatrix.size());
    if (numRows == 0) {
      return;
    }
    const int numCols = static_cast<int>(denseMatrix[0].size());

    rowPtr.push_back(0);
    for (int i = 0; i < numRows; ++i) {
      for (int j = 0; j < numCols; ++j) {
        if (denseMatrix[i][j] != T(0)) {
          colInd.push_back(j);
          values.push_back(denseMatrix[i][j]);
        }
      }
      rowPtr.push_back(static_cast<int>(colInd.size()));
    }
  }
};

std::vector<std::vector<float>> initializeDenseMatrix(int m, int n,
                                                      float sparsity) {
  std::vector<std::vector<float>> matrix(m, std::vector<float>(n, 0.0f));
  std::mt19937 rng(42);  // Fixed seed for reproducibility
  std::uniform_real_distribution<float> dist_val(0.0f, 1.0f);
  std::uniform_real_distribution<float> dist_prob(0.0f, 1.0f);

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      if (dist_prob(rng) > sparsity) {
        matrix[i][j] = dist_val(rng);  // Assign non-zero
      }
    }
  }
  return matrix;
}

// One thread per row ("CSR scalar").
__global__ void spmv_csr_kernel(int numRows, const int* rowPtr,
                                const int* colInd, const float* values,
                                const float* x, float* y) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < numRows) {
    float sum = 0.0f;
    for (int j = rowPtr[row]; j < rowPtr[row + 1]; ++j) {
      sum += values[j] * x[colInd[j]];
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
  const CSRMatrix<float> csr(dense);
  const int nnz = static_cast<int>(csr.values.size());
  printf("CSR SpMV: %d x %d matrix with %d non-zeros\n", rows, cols, nnz);

  const std::vector<float> h_x(cols, 1.12f);
  std::vector<float> h_y(rows);

  float *d_x, *d_y, *d_values;
  int *d_rowPtr, *d_colInd;
  CUDA_CHECK(cudaMalloc(&d_x, cols * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, rows * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_rowPtr, (rows + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_colInd, nnz * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_values, nnz * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), cols * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_rowPtr, csr.rowPtr.data(), (rows + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_colInd, csr.colInd.data(), nnz * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, csr.values.data(), nnz * sizeof(float),
                        cudaMemcpyHostToDevice));

  spmv_csr_kernel<<<lab::ceil_div(rows, BLOCK_SIZE), BLOCK_SIZE>>>(
      rows, d_rowPtr, d_colInd, d_values, d_x, d_y);
  CUDA_CHECK_LAUNCH();
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, rows * sizeof(float),
                        cudaMemcpyDeviceToHost));

  printf("First few results: %f %f %f %f %f\n", h_y[0], h_y[1], h_y[2], h_y[3],
         h_y[4]);
  const bool pass = lab::check_close("y = A x", h_y, spmv_reference(dense, h_x),
                                     1e-5, 1e-5);

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaFree(d_rowPtr));
  CUDA_CHECK(cudaFree(d_colInd));
  CUDA_CHECK(cudaFree(d_values));
  return lab::finish(pass);
}
