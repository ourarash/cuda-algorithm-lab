/*
 * cuSPARSE SpGEMM
 *
 * Intention:
 * This example shows how to multiply two sparse matrices with cuSPARSE's
 * SpGEMM API instead of writing a custom sparse matrix-matrix kernel.
 *
 * High-Level Algorithm:
 * - Generate two random sparse matrices and convert them to CSR.
 * - Upload both CSR matrices to the GPU.
 * - Ask cuSPARSE to estimate workspace, compute the sparse product, and
 *   materialize the result matrix C in CSR format.
 * - Convert C back to dense form on the host and validate against a dense CPU
 *   reference multiplication.
 */
#include <cuda_runtime.h>
#include <cusparse.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

bool floatEqual(float a, float b, float eps = 1e-3f) {
  return std::fabs(a - b) < eps * std::fmax(std::fabs(a), 1.0f);
}
// Dense reference: C = A × B
void denseMatMul(const std::vector<std::vector<float>> &A,
                 const std::vector<std::vector<float>> &B,
                 std::vector<std::vector<float>> &C) {
  int m = A.size(), k = A[0].size(), n = B[0].size();
  C.assign(m, std::vector<float>(n, 0.0f));
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      for (int p = 0; p < k; ++p) C[i][j] += A[i][p] * B[p][j];
}

// Multiply two sparse matrices using cuSPARSE's SpGEMM
// This function assumes the input matrices are in CSR format and stored on the
// GPU. It will output the result in CSR format as well. m, k, n are the
// dimensions of the matrices A (m x k) and B (k x n) nnzA and nnzB are the
// number of non-zero elements in A and B d_csrRowPtrA, d_csrColIndA, d_csrValA
// are the CSR representation of matrix A d_csrRowPtrB, d_csrColIndB, d_csrValB
// are the CSR representation of matrix B d_csrRowPtrC_out, d_csrColIndC_out,
// d_csrValC_out are pointers to the output CSR representation of matrix C
// nnzC_out is a pointer to the number of non-zero elements in the output matrix
// C.
void spgemm_example(int m, int k, int n, int nnzA, int *d_csrRowPtrA,
                    int *d_csrColIndA, float *d_csrValA, int nnzB,
                    int *d_csrRowPtrB, int *d_csrColIndB, float *d_csrValB,
                    int **d_csrRowPtrC_out, int **d_csrColIndC_out,
                    float **d_csrValC_out, int *nnzC_out) {
  // Scalars
  float alpha = 1.0f, beta = 0.0f;

  // cuSPARSE handle
  cusparseHandle_t handle;
  cusparseCreate(&handle);

  // Create sparse matrix descriptors
  cusparseSpMatDescr_t matA, matB, matC;
  cusparseCreateCsr(&matA, m, k, nnzA, d_csrRowPtrA, d_csrColIndA, d_csrValA,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  cusparseCreateCsr(&matB, k, n, nnzB, d_csrRowPtrB, d_csrColIndB, d_csrValB,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  cusparseCreateCsr(&matC, m, n, 0, nullptr, nullptr, nullptr,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

  // Create SpGEMM descriptor
  cusparseSpGEMMDescr_t spgemmDesc;
  cusparseSpGEMM_createDescr(&spgemmDesc);

  // Work estimation phase
  size_t bufferSize1 = 0, bufferSize2 = 0;
  void *dBuffer1 = nullptr, *dBuffer2 = nullptr;

  cusparseSpGEMM_workEstimation(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
      CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, nullptr);
  cudaMalloc(&dBuffer1, bufferSize1);
  cusparseSpGEMM_workEstimation(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
      CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, dBuffer1);

  // Compute phase
  cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB,
                         &beta, matC, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                         spgemmDesc, &bufferSize2, nullptr);
  cudaMalloc(&dBuffer2, bufferSize2);
  cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB,
                         &beta, matC, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                         spgemmDesc, &bufferSize2, dBuffer2);

  // Get output size
  int64_t C_num_rows, C_num_cols, C_nnz64;
  cusparseSpMatGetSize(matC, &C_num_rows, &C_num_cols, &C_nnz64);
  int C_nnz = static_cast<int>(C_nnz64);

  // Allocate output buffers
  float *d_csrValC;
  int *d_csrRowPtrC, *d_csrColIndC;
  cudaMalloc((void **)&d_csrRowPtrC, (m + 1) * sizeof(int));
  cudaMalloc((void **)&d_csrColIndC, C_nnz * sizeof(int));
  cudaMalloc((void **)&d_csrValC, C_nnz * sizeof(float));

  // Assign pointers to C matrix
  cusparseCsrSetPointers(matC, d_csrRowPtrC, d_csrColIndC, d_csrValC);

  // Copy final result
  cusparseSpGEMM_copy(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB,
                      &beta, matC, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                      spgemmDesc);

  // Set output pointers
  *d_csrRowPtrC_out = d_csrRowPtrC;
  *d_csrColIndC_out = d_csrColIndC;
  *d_csrValC_out = d_csrValC;
  *nnzC_out = C_nnz;

  // Cleanup
  cudaFree(dBuffer1);
  cudaFree(dBuffer2);
  cusparseSpGEMM_destroyDescr(spgemmDesc);
  cusparseDestroySpMat(matA);
  cusparseDestroySpMat(matB);
  cusparseDestroySpMat(matC);
  cusparseDestroy(handle);
}

// Convert CSR to dense for result verification
void csrToDense(int m, int n, const int *rowPtr, const int *colInd,
                const float *val, std::vector<std::vector<float>> &dense) {
  dense.assign(m, std::vector<float>(n, 0.0f));
  for (int i = 0; i < m; ++i) {
    for (int j = rowPtr[i]; j < rowPtr[i + 1]; ++j) {
      dense[i][colInd[j]] = val[j];
    }
  }
}

float benchmark_spgemm_with_validation(
    int numRuns, int m, int k, int n, int nnzA, int *d_csrRowPtrA,
    int *d_csrColIndA, float *d_csrValA, int nnzB, int *d_csrRowPtrB,
    int *d_csrColIndB, float *d_csrValB,
    const std::vector<std::vector<float>> &denseA,
    const std::vector<std::vector<float>> &denseB) {
  // 1. Run once for validation
  int *d_rowPtrC = nullptr, *d_colIndC = nullptr;
  float *d_valC = nullptr;
  int nnzC = 0;

  spgemm_example(m, k, n, nnzA, d_csrRowPtrA, d_csrColIndA, d_csrValA, nnzB,
                 d_csrRowPtrB, d_csrColIndB, d_csrValB, &d_rowPtrC, &d_colIndC,
                 &d_valC, &nnzC);

  std::vector<int> h_rowPtrC(m + 1), h_colIndC(nnzC);
  std::vector<float> h_valC(nnzC);
  cudaMemcpy(h_rowPtrC.data(), d_rowPtrC, (m + 1) * sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_colIndC.data(), d_colIndC, nnzC * sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_valC.data(), d_valC, nnzC * sizeof(float),
             cudaMemcpyDeviceToHost);

  // Convert result to dense
  std::vector<std::vector<float>> C_dense;
  csrToDense(m, n, h_rowPtrC.data(), h_colIndC.data(), h_valC.data(), C_dense);

  // Reference result
  std::vector<std::vector<float>> C_ref;
  denseMatMul(denseA, denseB, C_ref);

  // Validate
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      assert(floatEqual(C_dense[i][j], C_ref[i][j]));
    }
  }

  printf("✅ Validation passed!\n");

  // Free C once before benchmarking
  cudaFree(d_rowPtrC);
  cudaFree(d_colIndC);
  cudaFree(d_valC);

  // 2. Timed benchmark loop
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (int i = 0; i < numRuns; ++i) {
    int *d_rowPtrC = nullptr, *d_colIndC = nullptr;
    float *d_valC = nullptr;
    int nnzC = 0;

    spgemm_example(m, k, n, nnzA, d_csrRowPtrA, d_csrColIndA, d_csrValA, nnzB,
                   d_csrRowPtrB, d_csrColIndB, d_csrValB, &d_rowPtrC,
                   &d_colIndC, &d_valC, &nnzC);

    cudaFree(d_rowPtrC);
    cudaFree(d_colIndC);
    cudaFree(d_valC);
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return ms / numRuns;
}

void denseToCSR(const std::vector<std::vector<float>> &dense,
                std::vector<float> &csrVal, std::vector<int> &csrColInd,
                std::vector<int> &csrRowPtr) {
  int m = dense.size();
  int n = dense[0].size();

  csrVal.clear();
  csrColInd.clear();
  csrRowPtr.resize(m + 1);
  int nnz = 0;

  for (int i = 0; i < m; ++i) {
    csrRowPtr[i] = nnz;
    for (int j = 0; j < n; ++j) {
      if (dense[i][j] != 0.0f) {
        csrVal.push_back(dense[i][j]);
        csrColInd.push_back(j);
        ++nnz;
      }
    }
  }
  csrRowPtr[m] = nnz;
}

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

int main() {
  // Example usage of spgemm_example
  // Define matrices A and B in CSR format, allocate memory, and call the
  // function Note: Actual matrix data and sizes should be provided here

  const int m = 100;               // Number of rows in A
  const int k = 100;               // Number of columns in A and rows in B
  const int n = 100;               // Number of columns in B
  const int iteration_count = 10;  // Number of iterations for benchmarking

  std::vector<std::vector<float>> denseA = initializeDenseMatrix(m, k);

  // Matrtix A in CSR format
  std::vector<float> csrValA;
  std::vector<int> csrColIndA, csrRowPtrA;

  denseToCSR(denseA, csrValA, csrColIndA, csrRowPtrA);

  // Then upload to GPU
  float *d_valA;
  int *d_colIndA, *d_rowPtrA;
  int nnzA = csrValA.size();

  cudaMalloc(&d_valA, nnzA * sizeof(float));
  cudaMalloc(&d_colIndA, nnzA * sizeof(int));
  cudaMalloc(&d_rowPtrA, (denseA.size() + 1) * sizeof(int));

  cudaMemcpy(d_valA, csrValA.data(), nnzA * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_colIndA, csrColIndA.data(), nnzA * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_rowPtrA, csrRowPtrA.data(), (denseA.size() + 1) * sizeof(int),
             cudaMemcpyHostToDevice);

  // Matrix B in CSR format
  std::vector<std::vector<float>> denseB = initializeDenseMatrix(k, n);
  std::vector<float> csrValB;
  std::vector<int> csrColIndB, csrRowPtrB;

  denseToCSR(denseB, csrValB, csrColIndB, csrRowPtrB);

  // Then upload to GPU
  float *d_valB;
  int *d_colIndB, *d_rowPtrB;
  int nnzB = csrValB.size();

  cudaMalloc(&d_valB, nnzB * sizeof(float));
  cudaMalloc(&d_colIndB, nnzB * sizeof(int));
  cudaMalloc(&d_rowPtrB, (denseB.size() + 1) * sizeof(int));

  cudaMemcpy(d_valB, csrValB.data(), nnzB * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_colIndB, csrColIndB.data(), nnzB * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_rowPtrB, csrRowPtrB.data(), (denseB.size() + 1) * sizeof(int),
             cudaMemcpyHostToDevice);

  // Call the sparse matrix multiplication function
  float averageTime = benchmark_spgemm_with_validation(
      iteration_count, m, k, n, nnzA, d_rowPtrA, d_colIndA, d_valA, nnzB,
      d_rowPtrB, d_colIndB, d_valB, denseA, denseB);

  printf("Average time per run: %.2f ms\n", averageTime);

  // Free allocated memory
  cudaFree(d_valA);
  cudaFree(d_colIndA);
  cudaFree(d_rowPtrA);
  cudaFree(d_valB);
  cudaFree(d_colIndB);
  cudaFree(d_rowPtrB);

  return 0;
}
