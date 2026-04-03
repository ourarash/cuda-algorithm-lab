/*
 * Uncoalesced Matrix Multiplication
 *
 * Intention:
 * This file intentionally demonstrates a poor thread-to-data mapping so the
 * memory-access anti-pattern is easy to see.
 *
 * High-Level Algorithm:
 * - Launch a 2D thread block.
 * - Let each thread compute one output element C[i, j].
 * - Map threadIdx.x to rows instead of columns, which makes neighboring
 *   threads walk through memory with a large stride.
 * - Use this version as the baseline that later kernels improve upon.
 */
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "     \
                << cudaGetErrorString(err) << std::endl;                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))

/**
 * 0. The Anti-Pattern: Uncoalesced Memory Access
 * This kernel maps the fastest-changing thread index (threadIdx.x) to matrix
 * rows. Because memory is row-major, adjacent threads access memory locations
 * that are far apart, leading to a catastrophic loss in memory bandwidth.
 */
__global__ void sgemm_uncoalesced(int M, int N, int K, float alpha,
                                  const float *A, const float *B, float beta,
                                  float *C) {
  // compute position in C that this thread is responsible for
  // Note that i is changing with threadIdx.x and j with threadIdx.y
  // This means i is changing faster than j
  // so access of
  // - A[i, k] is not coalesced (adjacent threads access different rows)
  // - B[k, j] is broadcasted (since k, j are constant in the warp)
  // - C[i, j] is not coalesced (adjacent threads access different rows)
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint j = blockIdx.y * blockDim.y + threadIdx.y;

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (i < M && j < N) {
    float tmp = 0.0;
    for (int k = 0; k < K; ++k) {
      tmp += A[i * K + k] * B[k * N + j]; // A[i, k] * B[k, j]
    }
    // C = α*(A@B)+β*C
    C[i * N + j] = alpha * tmp + beta * C[i * N + j];
  }
}

void cpu_gemm(int M, int N, int K, float alpha, const float *A, const float *B,
              float beta, float *C) {
  for (int x = 0; x < M; ++x)
    for (int y = 0; y < N; ++y) {
      float tmp = 0.0f;
      for (int i = 0; i < K; ++i)
        tmp += A[x * K + i] * B[i * N + y];
      C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

bool nearly_equal(float a, float b, float eps = 1e-4f) {
  return std::fabs(a - b) < eps;
}

int main() {
  const int M = 1024, N = 1024, K = 1024;
  const int BLOCK_SIZE = 32;
  float alpha = 1.0f, beta = 0.0f;

  std::vector<float> A(M * K), B(K * N), C_cpu(M * N), C_gpu(M * N);

  for (int i = 0; i < M * K; ++i)
    A[i] = static_cast<float>(i % 13);
  for (int i = 0; i < K * N; ++i)
    B[i] = static_cast<float>((i % 7) - 3);
  for (int i = 0; i < M * N; ++i) {
    C_cpu[i] = 1.0f;
    C_gpu[i] = 1.0f;
  }

  std::cout << "Running CPU validation..." << std::endl;
  cpu_gemm(M, N, K, alpha, A.data(), B.data(), beta, C_cpu.data());

  float *dA, *dB, *dC;
  CHECK(cudaMalloc(&dA, A.size() * sizeof(float)));
  CHECK(cudaMalloc(&dB, B.size() * sizeof(float)));
  CHECK(cudaMalloc(&dC, C_gpu.size() * sizeof(float)));

  CHECK(cudaMemcpy(dA, A.data(), A.size() * sizeof(float),
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dB, B.data(), B.size() * sizeof(float),
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dC, C_gpu.data(), C_gpu.size() * sizeof(float),
                   cudaMemcpyHostToDevice));

  dim3 gridDim(CEIL_DIV(M, BLOCK_SIZE), CEIL_DIV(N, BLOCK_SIZE), 1);
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);

  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));

  CHECK(cudaEventRecord(start));
  sgemm_uncoalesced<<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
  CHECK(cudaEventRecord(stop));

  CHECK(cudaGetLastError());
  CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK(cudaEventElapsedTime(&ms, start, stop));
  std::cout << "Kernel Execution Time (Uncoalesced): " << ms << " ms\n";

  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));

  CHECK(cudaMemcpy(C_gpu.data(), dC, C_gpu.size() * sizeof(float),
                   cudaMemcpyDeviceToHost));

  bool pass = true;
  for (int i = 0; i < M * N; ++i) {
    if (!nearly_equal(C_cpu[i], C_gpu[i])) {
      std::cerr << "Mismatch at " << i << ": CPU=" << C_cpu[i]
                << ", GPU=" << C_gpu[i] << std::endl;
      pass = false;
      break;
    }
  }

  if (pass) {
    std::cout << "Validation PASSED!" << std::endl;
  }

  CHECK(cudaFree(dA));
  CHECK(cudaFree(dB));
  CHECK(cudaFree(dC));
  return 0;
}
