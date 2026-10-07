/*
 * cuBLAS GEMM
 *
 * Intention:
 * This example shows how to hand matrix multiplication to NVIDIA's tuned
 * cuBLAS library instead of writing a custom kernel, and how to use it with
 * the row-major matrices used everywhere else in this repo. Its GFLOP/s is
 * the baseline the hand-written kernels in matmul/ are measured against.
 *
 * High-Level Algorithm:
 * - Multiply two small row-major matrices with cublasSgemm and print C.
 * - Multiply two large random matrices, spot-check the result against the
 *   CPU, and report GFLOP/s.
 *
 * Row-major vs. column-major:
 * cuBLAS follows the Fortran BLAS convention and reads matrices column-major.
 * A row-major M x N matrix has exactly the same bytes as a column-major
 * N x M matrix, its transpose. So instead of C = A * B we ask cuBLAS for
 *   C^T = B^T * A^T
 * by swapping the operands and the M/N sizes. The column-major C^T it writes
 * is our row-major C. No data is moved or transposed.
 */
#include <cublas_v2.h>

#include <cstdio>
#include <vector>

#include "lab.cuh"

#define CUBLAS_CHECK(call)                                                 \
  do {                                                                     \
    cublasStatus_t status_ = (call);                                       \
    if (status_ != CUBLAS_STATUS_SUCCESS) {                                \
      std::fprintf(stderr, "cuBLAS error %d at %s:%d\n",                   \
                   static_cast<int>(status_), __FILE__, __LINE__);         \
      std::exit(EXIT_FAILURE);                                             \
    }                                                                      \
  } while (0)

// Row-major C (M x N) = A (M x K) * B (K x N) on device pointers.
void sgemm_row_major(cublasHandle_t handle, int M, int N, int K,
                     const float *A, const float *B, float *C) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  // Column-major view: C^T (N x M) = B^T (N x K) * A^T (K x M).
  // Leading dimensions are the row lengths of the row-major matrices.
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                           B, N, A, K, &beta, C, N));
}

// The 3x3 example from the comments: prints C so you can check it by hand.
bool small_example(cublasHandle_t handle) {
  const int n = 3;
  const std::vector<float> A = {1, 2, 3,
                                4, 5, 6,
                                7, 8, 9};
  const std::vector<float> B = {9, 8, 7,
                                6, 5, 4,
                                3, 2, 1};
  // A * B, computed by hand.
  const std::vector<float> expected = {30, 24, 18,
                                       84, 69, 54,
                                       138, 114, 90};

  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B, n * n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C, n * n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_A, A.data(), n * n * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, B.data(), n * n * sizeof(float),
                        cudaMemcpyHostToDevice));

  sgemm_row_major(handle, n, n, n, d_A, d_B, d_C);

  std::vector<float> C(n * n);
  CUDA_CHECK(cudaMemcpy(C.data(), d_C, n * n * sizeof(float),
                        cudaMemcpyDeviceToHost));
  printf("Result matrix C = A * B (row-major):\n");
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      printf("%6.1f ", C[i * n + j]);
    }
    printf("\n");
  }

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  return lab::check_close("3x3 C", C, expected, 0.0, 1e-4);
}

int main(int argc, char **argv) {
  lab::Args args(argc, argv);
  const int size = static_cast<int>(args.get_int("n", args.quick() ? 257 : 4096));
  const int M = size, N = size, K = size;

  lab::print_device();
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  bool pass = small_example(handle);

  // Large random problem for timing.
  printf("\ncublasSgemm with M=N=K=%d\n", size);
  const std::vector<float> A =
      lab::random_uniform<float>(static_cast<size_t>(M) * K, -1.f, 1.f, 1);
  const std::vector<float> B =
      lab::random_uniform<float>(static_cast<size_t>(K) * N, -1.f, 1.f, 2);
  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, A.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B, B.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C, static_cast<size_t>(M) * N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_A, A.data(), A.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, B.data(), B.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  sgemm_row_major(handle, M, N, K, d_A, d_B, d_C);
  std::vector<float> C(static_cast<size_t>(M) * N);
  CUDA_CHECK(cudaMemcpy(C.data(), d_C, C.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // A full CPU reference would take minutes at this size, so check a sample
  // of entries, each against a double-precision dot product.
  std::vector<float> sampled;
  std::vector<double> expected;
  for (int s = 0; s < 512; ++s) {
    const int i = (s * 7919) % M;
    const int j = (s * 104729) % N;
    double dot = 0.0;
    for (int k = 0; k < K; ++k) {
      dot += static_cast<double>(A[static_cast<size_t>(i) * K + k]) *
             B[static_cast<size_t>(k) * N + j];
    }
    sampled.push_back(C[static_cast<size_t>(i) * N + j]);
    expected.push_back(dot);
  }
  // cuBLAS may use TF32 or other reduced-precision paths only if asked to,
  // so plain FP32 accuracy applies here.
  pass = lab::check_close("sampled C entries", sampled, expected, 1e-4, 1e-3) &&
         pass;

  const float ms =
      lab::time_ms([&] { sgemm_row_major(handle, M, N, K, d_A, d_B, d_C); });
  lab::report("cublasSgemm", ms, 2.0 * M * N * K, /*bytes=*/0);

  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  return lab::finish(pass);
}
