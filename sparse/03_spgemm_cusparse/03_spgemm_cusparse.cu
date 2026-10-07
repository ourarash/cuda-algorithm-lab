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
 *
 * SpGEMM is a two-phase computation because the number of non-zeros in C is
 * not known in advance: cuSPARSE first estimates work and computes C's
 * structure, then the caller allocates C, and only then are the values
 * copied out.
 */
#include <cusparse.h>

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "lab.cuh"

#define CUSPARSE_CHECK(call)                                             \
  do {                                                                   \
    cusparseStatus_t status_ = (call);                                   \
    if (status_ != CUSPARSE_STATUS_SUCCESS) {                            \
      std::fprintf(stderr, "cuSPARSE error %s at %s:%d\n",               \
                   cusparseGetErrorString(status_), __FILE__, __LINE__); \
      std::exit(EXIT_FAILURE);                                           \
    }                                                                    \
  } while (0)

// Dense reference in double precision: C = A × B
std::vector<double> denseMatMul(const std::vector<std::vector<float>>& A,
                                const std::vector<std::vector<float>>& B) {
  const size_t m = A.size(), k = A[0].size(), n = B[0].size();
  std::vector<double> C(m * n, 0.0);
  for (size_t i = 0; i < m; ++i)
    for (size_t p = 0; p < k; ++p)
      for (size_t j = 0; j < n; ++j)
        C[i * n + j] += static_cast<double>(A[i][p]) * B[p][j];
  return C;
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
void spgemm_example(int m, int k, int n, int nnzA, int* d_csrRowPtrA,
                    int* d_csrColIndA, float* d_csrValA, int nnzB,
                    int* d_csrRowPtrB, int* d_csrColIndB, float* d_csrValB,
                    int** d_csrRowPtrC_out, int** d_csrColIndC_out,
                    float** d_csrValC_out, int* nnzC_out) {
  // Scalars
  float alpha = 1.0f, beta = 0.0f;

  // cuSPARSE handle
  cusparseHandle_t handle;
  CUSPARSE_CHECK(cusparseCreate(&handle));

  // Create sparse matrix descriptors
  cusparseSpMatDescr_t matA, matB, matC;
  CUSPARSE_CHECK(cusparseCreateCsr(&matA, m, k, nnzA, d_csrRowPtrA,
                                   d_csrColIndA, d_csrValA, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                   CUDA_R_32F));
  CUSPARSE_CHECK(cusparseCreateCsr(&matB, k, n, nnzB, d_csrRowPtrB,
                                   d_csrColIndB, d_csrValB, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                   CUDA_R_32F));
  CUSPARSE_CHECK(cusparseCreateCsr(&matC, m, n, 0, nullptr, nullptr, nullptr,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

  // Create SpGEMM descriptor
  cusparseSpGEMMDescr_t spgemmDesc;
  CUSPARSE_CHECK(cusparseSpGEMM_createDescr(&spgemmDesc));

  // Work estimation phase
  size_t bufferSize1 = 0, bufferSize2 = 0;
  void *dBuffer1 = nullptr, *dBuffer2 = nullptr;

  CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
      CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, nullptr));
  CUDA_CHECK(cudaMalloc(&dBuffer1, bufferSize1));
  CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
      CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, dBuffer1));

  // Compute phase
  CUSPARSE_CHECK(cusparseSpGEMM_compute(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
      CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, nullptr));
  CUDA_CHECK(cudaMalloc(&dBuffer2, bufferSize2));
  CUSPARSE_CHECK(cusparseSpGEMM_compute(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
      CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, dBuffer2));

  // Get output size
  int64_t C_num_rows, C_num_cols, C_nnz64;
  CUSPARSE_CHECK(
      cusparseSpMatGetSize(matC, &C_num_rows, &C_num_cols, &C_nnz64));
  int C_nnz = static_cast<int>(C_nnz64);

  // Allocate output buffers
  float* d_csrValC;
  int *d_csrRowPtrC, *d_csrColIndC;
  CUDA_CHECK(cudaMalloc((void**)&d_csrRowPtrC, (m + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&d_csrColIndC, C_nnz * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&d_csrValC, C_nnz * sizeof(float)));

  // Assign pointers to C matrix
  CUSPARSE_CHECK(
      cusparseCsrSetPointers(matC, d_csrRowPtrC, d_csrColIndC, d_csrValC));

  // Copy final result
  CUSPARSE_CHECK(cusparseSpGEMM_copy(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                     matA, matB, &beta, matC, CUDA_R_32F,
                                     CUSPARSE_SPGEMM_DEFAULT, spgemmDesc));

  // Set output pointers
  *d_csrRowPtrC_out = d_csrRowPtrC;
  *d_csrColIndC_out = d_csrColIndC;
  *d_csrValC_out = d_csrValC;
  *nnzC_out = C_nnz;

  // Cleanup
  CUDA_CHECK(cudaFree(dBuffer1));
  CUDA_CHECK(cudaFree(dBuffer2));
  CUSPARSE_CHECK(cusparseSpGEMM_destroyDescr(spgemmDesc));
  CUSPARSE_CHECK(cusparseDestroySpMat(matA));
  CUSPARSE_CHECK(cusparseDestroySpMat(matB));
  CUSPARSE_CHECK(cusparseDestroySpMat(matC));
  CUSPARSE_CHECK(cusparseDestroy(handle));
}

// Convert CSR to dense for result verification
void csrToDense(int m, int n, const int* rowPtr, const int* colInd,
                const float* val, std::vector<std::vector<float>>& dense) {
  dense.assign(m, std::vector<float>(n, 0.0f));
  for (int i = 0; i < m; ++i) {
    for (int j = rowPtr[i]; j < rowPtr[i + 1]; ++j) {
      dense[i][colInd[j]] = val[j];
    }
  }
}

float benchmark_spgemm_with_validation(
    bool* pass, int numRuns, int m, int k, int n, int nnzA, int* d_csrRowPtrA,
    int* d_csrColIndA, float* d_csrValA, int nnzB, int* d_csrRowPtrB,
    int* d_csrColIndB, float* d_csrValB,
    const std::vector<std::vector<float>>& denseA,
    const std::vector<std::vector<float>>& denseB) {
  // 1. Run once for validation
  int *d_rowPtrC = nullptr, *d_colIndC = nullptr;
  float* d_valC = nullptr;
  int nnzC = 0;

  spgemm_example(m, k, n, nnzA, d_csrRowPtrA, d_csrColIndA, d_csrValA, nnzB,
                 d_csrRowPtrB, d_csrColIndB, d_csrValB, &d_rowPtrC, &d_colIndC,
                 &d_valC, &nnzC);

  std::vector<int> h_rowPtrC(m + 1), h_colIndC(nnzC);
  std::vector<float> h_valC(nnzC);
  CUDA_CHECK(cudaMemcpy(h_rowPtrC.data(), d_rowPtrC, (m + 1) * sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_colIndC.data(), d_colIndC, nnzC * sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_valC.data(), d_valC, nnzC * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // Convert result to dense
  std::vector<std::vector<float>> C_dense;
  csrToDense(m, n, h_rowPtrC.data(), h_colIndC.data(), h_valC.data(), C_dense);

  // Validate against the dense reference.
  std::vector<float> C_flat;
  for (const auto& row : C_dense) {
    C_flat.insert(C_flat.end(), row.begin(), row.end());
  }
  printf("nnz(A) = %d, nnz(B) = %d, nnz(C) = %d\n", nnzA, nnzB, nnzC);
  *pass = lab::check_close("C = A B", C_flat, denseMatMul(denseA, denseB), 1e-5,
                           1e-5);

  // Free C once before benchmarking
  CUDA_CHECK(cudaFree(d_rowPtrC));
  CUDA_CHECK(cudaFree(d_colIndC));
  CUDA_CHECK(cudaFree(d_valC));

  // 2. Timed benchmark loop. Each iteration repeats the whole call, including
  // creating the handle and descriptors and allocating buffers, so this is
  // end-to-end latency rather than the cost of the multiplication alone.
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  for (int i = 0; i < numRuns; ++i) {
    int *d_rowPtrC = nullptr, *d_colIndC = nullptr;
    float* d_valC = nullptr;
    int nnzC = 0;

    spgemm_example(m, k, n, nnzA, d_csrRowPtrA, d_csrColIndA, d_csrValA, nnzB,
                   d_csrRowPtrB, d_csrColIndB, d_csrValB, &d_rowPtrC,
                   &d_colIndC, &d_valC, &nnzC);

    CUDA_CHECK(cudaFree(d_rowPtrC));
    CUDA_CHECK(cudaFree(d_colIndC));
    CUDA_CHECK(cudaFree(d_valC));
  }

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return ms / numRuns;
}

void denseToCSR(const std::vector<std::vector<float>>& dense,
                std::vector<float>& csrVal, std::vector<int>& csrColInd,
                std::vector<int>& csrRowPtr) {
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
                                                      unsigned seed,
                                                      float sparsity = 0.95f) {
  std::vector<std::vector<float>> matrix(m, std::vector<float>(n, 0.0f));
  std::mt19937 rng(seed);  // Fixed seed for reproducibility
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
  lab::print_device();

  const int m = 100;               // Number of rows in A
  const int k = 100;               // Number of columns in A and rows in B
  const int n = 100;               // Number of columns in B
  const int iteration_count = 10;  // Number of iterations for benchmarking

  std::vector<std::vector<float>> denseA =
      initializeDenseMatrix(m, k, /*seed=*/42);

  // Matrtix A in CSR format
  std::vector<float> csrValA;
  std::vector<int> csrColIndA, csrRowPtrA;

  denseToCSR(denseA, csrValA, csrColIndA, csrRowPtrA);

  // Then upload to GPU
  float* d_valA;
  int *d_colIndA, *d_rowPtrA;
  int nnzA = csrValA.size();

  CUDA_CHECK(cudaMalloc(&d_valA, nnzA * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_colIndA, nnzA * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_rowPtrA, (denseA.size() + 1) * sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_valA, csrValA.data(), nnzA * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_colIndA, csrColIndA.data(), nnzA * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_rowPtrA, csrRowPtrA.data(),
                        (denseA.size() + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));

  // Matrix B in CSR format
  std::vector<std::vector<float>> denseB =
      initializeDenseMatrix(k, n, /*seed=*/43);
  std::vector<float> csrValB;
  std::vector<int> csrColIndB, csrRowPtrB;

  denseToCSR(denseB, csrValB, csrColIndB, csrRowPtrB);

  // Then upload to GPU
  float* d_valB;
  int *d_colIndB, *d_rowPtrB;
  int nnzB = csrValB.size();

  CUDA_CHECK(cudaMalloc(&d_valB, nnzB * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_colIndB, nnzB * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_rowPtrB, (denseB.size() + 1) * sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_valB, csrValB.data(), nnzB * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_colIndB, csrColIndB.data(), nnzB * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_rowPtrB, csrRowPtrB.data(),
                        (denseB.size() + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));

  // Call the sparse matrix multiplication function
  bool pass = false;
  float averageTime = benchmark_spgemm_with_validation(
      &pass, iteration_count, m, k, n, nnzA, d_rowPtrA, d_colIndA, d_valA, nnzB,
      d_rowPtrB, d_colIndB, d_valB, denseA, denseB);

  printf("Average end-to-end time per SpGEMM call: %.3f ms\n", averageTime);

  // Free allocated memory
  CUDA_CHECK(cudaFree(d_valA));
  CUDA_CHECK(cudaFree(d_colIndA));
  CUDA_CHECK(cudaFree(d_rowPtrA));
  CUDA_CHECK(cudaFree(d_valB));
  CUDA_CHECK(cudaFree(d_colIndB));
  CUDA_CHECK(cudaFree(d_rowPtrB));

  return lab::finish(pass);
}
