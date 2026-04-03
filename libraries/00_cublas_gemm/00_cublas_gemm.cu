/*
 * cuBLAS GEMM
 *
 * Intention:
 * This example shows the simplest way to hand matrix multiplication over to
 * NVIDIA's tuned cuBLAS library instead of writing a custom kernel.
 *
 * High-Level Algorithm:
 * - Allocate three small matrices.
 * - Copy A and B to the GPU.
 * - Create a cuBLAS handle and call cublasSgemm.
 * - Copy the result matrix C back and print it.
 */
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))

int main() {
  const int N = 3;
  float A[N * N] = {1, 2, 3,
                    4, 5, 6,
                    7, 8, 9};
  float B[N * N] = {9, 8, 7,
                    6, 5, 4,
                    3, 2, 1};
  float C[N * N] = {0};

  float *d_A, *d_B, *d_C;
  cudaMalloc((void**)&d_A, N * N * sizeof(float));
  cudaMalloc((void**)&d_B, N * N * sizeof(float));
  cudaMalloc((void**)&d_C, N * N * sizeof(float));

  cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  cublasCreate(&handle);

  float alpha = 1.0f, beta = 0.0f;
  // cuBLAS expects column-major inputs by default.
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B,
              N, &beta, d_C, N);

  cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

  printf("Result matrix C:\n");
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      printf("%6.1f ", C[IDX2C(i, j, N)]);
    }
    printf("\n");
  }

  cublasDestroy(handle);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}
