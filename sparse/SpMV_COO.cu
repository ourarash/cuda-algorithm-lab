#include <cuda_runtime.h>

#include <iostream>
#include <vector>

template <typename T>
class CooMatrix {
 public:
  std::vector<int> rowInd;
  std::vector<int> colInd;
  std::vector<T> values;

  CooMatrix(std::vector<std::vector<T>>& denseMatrix) {
    denseToCOO(denseMatrix, rowInd, colInd, values);
  }

  void denseToCOO(const std::vector<std::vector<T>>& denseMatrix,
                  std::vector<int>& rowInd, std::vector<int>& colInd,
                  std::vector<T>& values) {
    rowInd.clear();
    colInd.clear();
    values.clear();

    int numRows = denseMatrix.size();
    if (numRows == 0) {
      return;
    }
    int numCols = denseMatrix[0].size();

    for (int i = 0; i < numRows; ++i) {
      for (int j = 0; j < numCols; ++j) {
        if (denseMatrix[i][j] != T(0)) {
          rowInd.push_back(i);
          colInd.push_back(j);
          values.push_back(denseMatrix[i][j]);
        }
      }
    }

    return {rowInd, colInd, values};
  }

  std::vector<std::vector<T>> CooToDense(const std::vector<int>& rowInd,
                                         const std::vector<int>& colInd,
                                         const std::vector<T>& values,
                                         int numRows, int numCols) {
    std::vector<std::vector<T>> denseMatrix(numRows,
                                            std::vector<T>(numCols, T(0)));

    for (size_t i = 0; i < rowInd.size(); ++i) {
      int row = rowInd[i];
      int col = colInd[i];
      denseMatrix[row][col] = values[i];
    }

    return denseMatrix;
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

__global__ void SpMV_kernel(int numRows, int numCols, int numNonZero,
                            const int* rowInd, const int* colInd,
                            const float* values, const float* in_vec,
                            float* out_vec) {
  int none_zero_item_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (none_zero_item_index < numNonZero) {
    int row = rowInd[none_zero_item_index];
    int col = colInd[none_zero_item_index];
    _atomicAdd(&out_vec[row], values[none_zero_item_index] * in_vec[col]);
  }
}

#define BLOCK_SIZE 256

void SpMV(const std::vector<std::vector<float>>& denseMatrix,
          const std::vector<float>& in_vec, std::vector<float>& out_vec) {
  CooMatrix<float> cooMatrix(denseMatrix);

  float *d_in_vec, *d_out_vec;
  int *d_rowInd, *d_colInd;
  CHECK(cudaMalloc(&d_in_vec, in_vec.size() * sizeof(float)));
  CHECK(cudaMalloc(&d_out_vec, denseMatrix.size() * sizeof(float)));
  CHECK(cudaMalloc(&d_rowInd, cooMatrix.rowInd.size() * sizeof(int)));
  CHECK(cudaMalloc(&d_colInd, cooMatrix.colInd.size() * sizeof(int)));

  CHECK(cudaMemcpy(d_in_vec, in_vec.data(), in_vec.size() * sizeof(float),
                   cudaMemcpyHostToDevice));

  CHECK(cudaMemcpy(d_rowInd, cooMatrix.rowInd.data(),
                   cooMatrix.rowInd.size() * sizeof(int),
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_colInd, cooMatrix.colInd.data(),
                   cooMatrix.colInd.size() * sizeof(int),
                   cudaMemcpyHostToDevice));

  SpMV_kernel<<<(cooMatrix.rowInd.size() + BLOCK_SIZE - 1) / BLOCK_SIZE,
                BLOCK_SIZE>>>(cooMatrix.rowInd.size(), denseMatrix[0].size(),
                              cooMatrix.values.size(), d_rowInd, d_colInd,
                              d_values, d_in_vec, d_out_vec);

  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(out_vec.data(), d_out_vec,
                   denseMatrix.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(d_in_vec));
  CHECK(cudaFree(d_out_vec));
  CHECK(cudaFree(d_rowInd));
  CHECK(cudaFree(d_colInd));
  CHECK(cudaFree(d_values));
}

int main() {
  auto denseMatrix =
      initializeDenseMatrix(/*m*/ 1000, /*n*/ 1000, /*sparsity*/ 0.99f);
  
      std::vector<float> in_vec = initializeVector(denseMatrix[0].size(), 1.12f);

  std::vector<float> out_vec(denseMatrix.size(), 0.0f);

  SpMV(denseMatrix, in_vec, out_vec);

  // Print first few results
  std::cout << "First few results of SpMV: ";
  for (int i = 0; i < 5 && i < out_vec.size(); ++i) {
    std::cout << out_vec[i] << " ";
  }
  std::cout << std::endl;

  return 0;
}
