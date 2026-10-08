/*
 * SpMV 7: cuSPARSE (the baseline)
 *
 * Intention:
 * cusparseSpMV with a CSR matrix descriptor, the library baseline for
 * steps 04-06. The generic API needs descriptors for the sparse matrix and
 * the dense vectors, and a workspace whose size cusparseSpMV_bufferSize
 * reports. Everything except the multiplication itself is created once in
 * setup().
 */
#include <cusparse.h>

#include "../spmv_harness.cuh"

#define CUSPARSE_CHECK(call)                                              \
  do {                                                                    \
    cusparseStatus_t status_ = (call);                                    \
    if (status_ != CUSPARSE_STATUS_SUCCESS) {                             \
      std::fprintf(stderr, "cuSPARSE error %s at %s:%d\n",                \
                   cusparseGetErrorString(status_), __FILE__, __LINE__);  \
      std::exit(EXIT_FAILURE);                                            \
    }                                                                     \
  } while (0)

static cusparseHandle_t handle;
static cusparseSpMatDescr_t matA;
static CsrDevice A;
static void *d_buffer = nullptr;
static size_t buffer_bytes = 0;

void setup(const CsrHost &, const CsrDevice &device) {
  A = device;
  CUSPARSE_CHECK(cusparseCreate(&handle));
  CUSPARSE_CHECK(cusparseCreateCsr(&matA, A.rows, A.cols, A.nnz, A.row_ptr,
                                   A.col_idx, A.vals, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                   CUDA_R_32F));
}

void run(const float *x, float *y) {
  const float alpha = 1.0f, beta = 0.0f;
  cusparseDnVecDescr_t vecX, vecY;
  CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, A.cols, const_cast<float *>(x), CUDA_R_32F));
  CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, A.rows, y, CUDA_R_32F));
  if (d_buffer == nullptr) {
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                           CUSPARSE_SPMV_ALG_DEFAULT, &buffer_bytes));
    CUDA_CHECK(cudaMalloc(&d_buffer, std::max<size_t>(buffer_bytes, 1)));
  }
  CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
                              vecX, &beta, vecY, CUDA_R_32F,
                              CUSPARSE_SPMV_ALG_DEFAULT, d_buffer));
  CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
  CUSPARSE_CHECK(cusparseDestroyDnVec(vecY));
}

int main(int argc, char **argv) {
  const int result = run_spmv("7. cuSPARSE SpMV", argc, argv, setup, run);
  cusparseDestroySpMat(matA);
  cusparseDestroy(handle);
  cudaFree(d_buffer);
  return result;
}
