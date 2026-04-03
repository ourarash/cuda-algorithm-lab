/*
 * Tensor Core Matrix Multiplication
 *
 * Intention:
 * This file demonstrates how to move from custom CUDA cores to NVIDIA Tensor
 * Cores through the WMMA API.
 *
 * High-Level Algorithm:
 * - Assign each warp to one 16x16 output tile.
 * - Load 16x16 fragments of A and B into WMMA fragments.
 * - Let the hardware perform matrix multiply-accumulate on Tensor Cores.
 * - Store the accumulated FP32 output fragment back to global memory.
 */
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <iostream>
#include <vector>

using namespace nvcuda;

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
 * 6. Hardware Acceleration (WMMA API / Tensor Cores)
 * This version targets NVIDIA's specialized Tensor Cores. It uses the Warp Matrix 
 * Multiply-Accumulate (WMMA) API to program an entire warp (32 threads) to natively
 * execute mixed-precision (FP16 inputs to FP32 accumulate) matrix operations.
 */
__global__ void sgemm_wmma(int M, int N, int K, float alpha, const half *A,
                         const half *B, float beta, float *C) {
  // WMMA fragment dimensions
  const int WMMA_M = 16;
  const int WMMA_N = 16;
  const int WMMA_K = 16;

  // Identify which tile of C this warp is computing
  int tileRow = blockIdx.y; // row tile index of C
  int tileCol = blockIdx.x; // col tile index of C

  // Declare fragments (register-level tiles distributed across the threads in a warp)
  // matrix_a and matrix_b store inputs (FP16), accumulator stores the result (FP32)
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

  // Initialize output fragment to zero
  wmma::fill_fragment(c_frag, 0.0f);

  // Loop over K dimension in chunks of WMMA_K
  for (int k = 0; k < K; k += WMMA_K) {
    // Compute base indices in A and B for this tile
    int a_row = tileRow * WMMA_M;
    int a_col = k;
    int b_row = k;
    int b_col = tileCol * WMMA_N;

    // Bounds checking before loading
    if (a_row < M && a_col < K && b_row < K && b_col < N) {
      // Load 16x16 tiles from global memory directly into the hardware fragments
      wmma::load_matrix_sync(a_frag, A + a_row * K + a_col, K);
      wmma::load_matrix_sync(b_frag, B + b_row * N + b_col, N);

      // Perform hardware-accelerated matrix multiply-accumulate: C += A * B
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
  }

  // --- Store the output fragment to global memory ---
  int c_row = tileRow * WMMA_M;
  int c_col = tileCol * WMMA_N;

  if (c_row < M && c_col < N) {
    // Load existing C tile into a fragment for the beta scaling
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> existing_c_frag;
    wmma::load_matrix_sync(existing_c_frag, C + c_row * N + c_col, N, wmma::mem_row_major);

    // Apply α and β scaling to each element in the fragment
    // C = α*(A@B)+β*C
    for(int i=0; i<existing_c_frag.num_elements; i++) {
        existing_c_frag.x[i] = alpha * c_frag.x[i] + beta * existing_c_frag.x[i];
    }
    
    // Store the final computed fragment back to global memory
    wmma::store_matrix_sync(C + c_row * N + c_col, existing_c_frag, N, wmma::mem_row_major);
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

bool nearly_equal(float a, float b, float eps = 1e-3f) { // Higher tolerance for mixed-precision
  return std::fabs(a - b) < eps;
}

int main() {
  const int M = 1024, N = 1024, K = 1024;
  const int WMMA_M = 16, WMMA_N = 16;
  float alpha = 1.0f, beta = 0.0f;

  std::vector<float> A(M * K), B(K * N), C_cpu(M * N), C_gpu(M * N);
  std::vector<half> A_half(M * K), B_half(K * N);

  for (int i = 0; i < M * K; ++i) A[i] = static_cast<float>(i % 13);
  for (int i = 0; i < K * N; ++i) B[i] = static_cast<float>((i % 7) - 3);
  for (int i = 0; i < M * N; ++i) {
    C_cpu[i] = 1.0f;
    C_gpu[i] = 1.0f;
  }

  // Convert float to half for GPU input
  for(int i=0; i < M*K; ++i) A_half[i] = __float2half(A[i]);
  for(int i=0; i < K*N; ++i) B_half[i] = __float2half(B[i]);

  std::cout << "Running CPU validation..." << std::endl;
  cpu_gemm(M, N, K, alpha, A.data(), B.data(), beta, C_cpu.data());

  half *dA, *dB;
  float *dC;
  CHECK(cudaMalloc(&dA, A_half.size() * sizeof(half)));
  CHECK(cudaMalloc(&dB, B_half.size() * sizeof(half)));
  CHECK(cudaMalloc(&dC, C_gpu.size() * sizeof(float)));

  CHECK(cudaMemcpy(dA, A_half.data(), A_half.size() * sizeof(half), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dB, B_half.data(), B_half.size() * sizeof(half), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dC, C_gpu.data(), C_gpu.size() * sizeof(float), cudaMemcpyHostToDevice));

  dim3 gridDim(CEIL_DIV(N, WMMA_N), CEIL_DIV(M, WMMA_M), 1);
  dim3 blockDim(32, 8, 1); // 256 threads, 8 warps

  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));

  CHECK(cudaEventRecord(start));
  sgemm_wmma<<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
  CHECK(cudaEventRecord(stop));

  CHECK(cudaGetLastError());
  CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK(cudaEventElapsedTime(&ms, start, stop));
  std::cout << "Kernel Execution Time (Tensor Cores): " << ms << " ms\n";

  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));

  CHECK(cudaMemcpy(C_gpu.data(), dC, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

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
