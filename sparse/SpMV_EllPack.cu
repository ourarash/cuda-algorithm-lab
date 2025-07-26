// This file is incompelte!
#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

#define CHECK(call)                                                         \
  do {                                                                      \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess) {                                               \
      std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " - " \
                << cudaGetErrorString(err) << std::endl;                    \
      exit(EXIT_FAILURE);                                                   \
    }                                                                       \
  } while (0)

// Ellpack matrix class definition
// We pad rows to the maximum number of non-zero elements in any row
// to ensure that all rows have the same length, then store padded array of
// nonzeros in column-major order.
template <typename T>
class EllpackMatrix {
 public:
  EllpackMatrix(const std::vector<std::vector<T>>& denseMatrix) {
    denseToEllpack(denseMatrix, rowPtr, colInd, values);
  }
  void denseToEllpack(const std::vector<std::vector<T>>& denseMatrix,
                      std::vector<int>& rowPtr, std::vector<int>& colInd,
                      std::vector<T>& values) {
    rowPtr.clear();
    colInd.clear();
    values.clear();

    int numRows = denseMatrix.size();
    if (numRows == 0) {
      return;
    }
    int maxNonZero = 0;

    // Find the maximum number of non-zero elements in any row
    for (const auto& row : denseMatrix) {
      int nonZeroCount = std::count_if(row.begin(), row.end(),
                                       [](T val) { return val != T(0); });
      maxNonZero = std::max(maxNonZero, nonZeroCount);
    }

    // Resize vectors to accommodate the maximum number of non-zero elements
    rowPtr.resize(numRows + 1);
    rowPtr[0] = 0;

    for (int i = 0; i < numRows; ++i) {
      for (int j = 0; j < maxNonZero; ++j) {
        if (j < denseMatrix[i].size() && denseMatrix[i][j] != T(0)) {
          colInd.push_back(j);
          values.push_back(denseMatrix[i][j]);
        } else {
          colInd.push_back(-1);    // Use -1 to indicate padding
          values.push_back(T(0));  // Use zero for padding
        }
      }
      rowPtr[i + 1] = rowPtr[i] + maxNonZero;
    }
  }
};

std::vector<std::vector<float>> initializeDenseMatrix(int m, int n,
                                                      float sparsity = 0.99f) {
  std::vector<std::vector<float>> matrix(m, std::vector<float>(n, 0.0f));
  std::mt19937 rng(42);  // Fixed seed for reproducibility
  std::uniform_real_distribution<float> dist_val(0.0f, 1.0f);
  std::uniform_real_distribution<float> dist_prob(0.0f, 1.0f);

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      if (dist_prob(rng) > sparsity) {
        matrix[i][j] = dist_val(rng);  // Assign non-zero
      }
      // else keep zero
    }
  }
  return matrix;
}

std::vector<float> initializeVector(int size, float value = 0.1f) {
  return std::vector<float>(size, value);
}

// We assign one thread per row.
__global__ void SpMV_kernel(int numRows, int numNonZero, const int* rowInd,
                            const int* colInd, const float* values,
                            const float* in_vec, float* out_vec) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < numRows) {
    float sum = 0.0f;
    for (int j = rowInd[row]; j < rowInd[row + 1]; ++j) {
      sum += values[j] * in_vec[colInd[j]];
    }
    out_vec[row] = sum;
  }
}

#define BLOCK_SIZE 256

void SpMV_gpu(const std::vector<std::vector<float>>& denseMatrix,
              const std::vector<float>& in_vec, std::vector<float>& out_vec) {
  CSRMatrix<float> csrMatrix(denseMatrix);

  float *d_in_vec, *d_out_vec;
  int *d_rowPtr, *d_colInd;
  float* d_values;
  CHECK(cudaMalloc(&d_in_vec, in_vec.size() * sizeof(float)));
  CHECK(cudaMalloc(&d_out_vec, denseMatrix.size() * sizeof(float)));
  CHECK(cudaMalloc(&d_rowPtr, csrMatrix.rowPtr.size() * sizeof(int)));
  CHECK(cudaMalloc(&d_colInd, csrMatrix.colInd.size() * sizeof(int)));
  CHECK(cudaMalloc(&d_values, csrMatrix.values.size() * sizeof(float)));

  CHECK(cudaMemcpy(d_in_vec, in_vec.data(), in_vec.size() * sizeof(float),
                   cudaMemcpyHostToDevice));

  // Initialize output vector to zero
  CHECK(cudaMemset(d_out_vec, 0, denseMatrix.size() * sizeof(float)));

  CHECK(cudaMemcpy(d_rowPtr, csrMatrix.rowPtr.data(),
                   csrMatrix.rowPtr.size() * sizeof(int),
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_colInd, csrMatrix.colInd.data(),
                   csrMatrix.colInd.size() * sizeof(int),
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_values, csrMatrix.values.data(),
                   csrMatrix.values.size() * sizeof(float),
                   cudaMemcpyHostToDevice));

  int numNonZero = csrMatrix.values.size();
  SpMV_kernel<<<(denseMatrix.size() + BLOCK_SIZE - 1) / BLOCK_SIZE,
                BLOCK_SIZE>>>(denseMatrix.size(), numNonZero, d_rowPtr,
                              d_colInd, d_values, d_in_vec, d_out_vec);

  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(out_vec.data(), d_out_vec,
                   denseMatrix.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(d_in_vec));
  CHECK(cudaFree(d_out_vec));
  CHECK(cudaFree(d_rowPtr));
  CHECK(cudaFree(d_colInd));
  CHECK(cudaFree(d_values));
}

std::vector<float> SpMV_cpu(const std::vector<std::vector<float>>& denseMatrix,
                            const std::vector<float>& in_vec) {
  std::vector<float> out_vec(denseMatrix.size(), 0.0f);
  for (size_t i = 0; i < denseMatrix.size(); ++i) {
    for (size_t j = 0; j < denseMatrix[i].size(); ++j) {
      out_vec[i] += denseMatrix[i][j] * in_vec[j];
    }
  }
  return out_vec;
}

int main() {
  auto denseMatrix =
      initializeDenseMatrix(/*m*/ 1000, /*n*/ 1000, /*sparsity*/ 0.99f);

  std::vector<float> in_vec = initializeVector(denseMatrix[0].size(), 1.12f);

  std::vector<float> out_vec(denseMatrix.size(), 0.0f);

  SpMV_gpu(denseMatrix, in_vec, out_vec);

  auto out_vec_cpu = SpMV_cpu(denseMatrix, in_vec);

  // Print first few results
  std::cout << "First few results of SpMV: ";
  for (int i = 0; i < 5 && i < out_vec.size(); ++i) {
    std::cout << out_vec[i] << " ";
  }
  std::cout << std::endl;

  // Verify results
  bool is_correct = true;
  for (size_t i = 0; i < out_vec.size(); ++i) {
    if (std::abs(out_vec[i] - out_vec_cpu[i]) > 1e-5f) {
      is_correct = false;
      break;
    }
  }
  if (is_correct) {
    std::cout << "✅ Results are correct!" << std::endl;
  } else {
    std::cout << "❌ Results are incorrect!" << std::endl;
  }

  return 0;
}
