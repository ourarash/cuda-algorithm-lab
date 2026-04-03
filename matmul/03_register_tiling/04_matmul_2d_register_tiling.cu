/*
 * 2D Register-Tiled Matrix Multiplication
 *
 * Intention:
 * This file extends register tiling so each thread computes a small 2D patch
 * of C instead of just a column vector.
 *
 * High-Level Algorithm:
 * - Stage A and B tiles in shared memory.
 * - Load a small register tile from both shared-memory tiles.
 * - Let each thread accumulate a TM x TN patch of output values in registers.
 * - Write the whole patch back to global memory at the end.
 */
#include <cassert>
#include <cmath>
#include <cstdlib>
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

// --- Tiling and Block Dimensions ---
constexpr int BM = 128; // Block size in M dimension
constexpr int BN = 128; // Block size in N dimension
constexpr int BK = 8;   // Inner loop tile size

// Work per thread (2D Register-level tiling)
// Each thread will compute an 8x8 grid of C.
constexpr int TM = 8;
constexpr int TN = 8;

// Thread block dimensions
// Number of threads = (128/8) * (128/8) = 16 * 16 = 256
constexpr int BLOCK_DIM_X = BN / TN;
constexpr int BLOCK_DIM_Y = BM / TM;

/**
 * 4. 2D Register Tiling
 * Building upon 1D register tiling, each thread now computes a 2D grid
 * (8x8) of output elements. It loads 8 elements from the A tile and 8
 * elements from the B tile into local registers, then performs 64
 * multiply-accumulate operations. This drastically improves arithmetic
 * intensity and heavily minimizes shared memory bottleneck.
 */
__global__ void sgemm_2d_register_tiling(int M, int N, int K, float alpha,
                                         const float *A, const float *B,
                                         float beta, float *C) {
  // Stage one K-slice of A and B for the whole thread block.
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  // Each thread is identified by its 2D position inside the 16x16 block.
  const int threadRow = threadIdx.y;
  const int threadCol = threadIdx.x;

  // This block is responsible for one 128x128 tile of the output matrix C.
  const int blockRow = blockIdx.y * BM;
  const int blockCol = blockIdx.x * BN;

  // Flattened thread id used to distribute cooperative shared-memory loads.
  const int threadId = threadIdx.y * blockDim.x + threadIdx.x;

  // Each thread accumulates an 8x8 output tile in registers.
  // Compared with 1D tiling, this reuses both loaded A values and loaded B
  // values across multiple FMAs per thread, which raises arithmetic
  // intensity and reduces shared-memory traffic.
  float threadResults[TM * TN] = {0.0f};
  float regM[TM] = {0.0f};
  float regN[TN] = {0.0f};

  // Sweep across K in chunks of BK. Every iteration multiplies one BMxBK
  // tile of A with one BKxBN tile of B.
  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // Cooperatively load the A tile into shared memory.
    // 256 threads fill BM*BK = 128*8 = 1024 elements, so each thread loads 4.
    for (int loadOffset = 0; loadOffset < BM * BK; loadOffset += 256) {
      int loadId = threadId + loadOffset;
      int a_row = loadId / BK;
      int a_col = loadId % BK;
      int a_global_row = blockRow + a_row;
      int a_global_col = bkIdx + a_col;

      if (a_global_row < M && a_global_col < K) {
        As[a_row][a_col] = A[a_global_row * K + a_global_col];
      } else {
        As[a_row][a_col] = 0.0f;
      }
    }

    // Cooperatively load the matching B tile into shared memory.
    for (int loadOffset = 0; loadOffset < BK * BN; loadOffset += 256) {
      int loadId = threadId + loadOffset;
      int b_row = loadId / BN;
      int b_col = loadId % BN;
      int b_global_row = bkIdx + b_row;
      int b_global_col = blockCol + b_col;

      if (b_global_row < K && b_global_col < N) {
        Bs[b_row][b_col] = B[b_global_row * N + b_global_col];
      } else {
        Bs[b_row][b_col] = 0.0f;
      }
    }

    // Make sure the whole block sees the fully populated shared tiles.
    __syncthreads();

    // Consume the staged tiles one K position at a time.
    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // Pull one column fragment from A and one row fragment from B into
      // registers. These fragments feed the full 8x8 outer product update.
      for (int i = 0; i < TM; ++i) {
        regM[i] = As[threadRow * TM + i][dotIdx];
      }
      for (int j = 0; j < TN; ++j) {
        regN[j] = Bs[dotIdx][threadCol * TN + j];
      }

      // Perform 64 FMAs in registers for this K step.
      // This is the key advantage over 1D tiling: one set of loaded register
      // values updates an 8x8 patch instead of only a row or a column, so the
      // thread does more math per shared-memory read.
      for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
          threadResults[i * TN + j] += regM[i] * regN[j];
        }
      }
    }

    // Wait until all threads are done before overwriting shared memory with
    // the next K tile.
    __syncthreads();
  }

  // Write each thread's 8x8 accumulated tile back to global memory.
  for (int i = 0; i < TM; ++i) {
    for (int j = 0; j < TN; ++j) {
      int c_row = blockRow + threadRow * TM + i;
      int c_col = blockCol + threadCol * TN + j;

      if (c_row < M && c_col < N) {
        C[c_row * N + c_col] =
            alpha * threadResults[i * TN + j] + beta * C[c_row * N + c_col];
      }
    }
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
  float alpha = 1.0f, beta = 0.0f;

  std::vector<float> A(M * K), B(K * N), C_cpu(M * N), C_gpu(M * N);

  for (int i = 0; i < M * K; ++i) A[i] = static_cast<float>(i % 13);
  for (int i = 0; i < K * N; ++i) B[i] = static_cast<float>((i % 7) - 3);
  for (int i = 0; i < M * N; ++i) { C_cpu[i] = 1.0f; C_gpu[i] = 1.0f; }

  std::cout << "Running CPU validation..." << std::endl;
  cpu_gemm(M, N, K, alpha, A.data(), B.data(), beta, C_cpu.data());

  float *dA, *dB, *dC;
  CHECK(cudaMalloc(&dA, A.size() * sizeof(float)));
  CHECK(cudaMalloc(&dB, B.size() * sizeof(float)));
  CHECK(cudaMalloc(&dC, C_gpu.size() * sizeof(float)));

  CHECK(cudaMemcpy(dA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dB, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dC, C_gpu.data(), C_gpu.size() * sizeof(float), cudaMemcpyHostToDevice));

  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1);
  dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y, 1);

  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));

  CHECK(cudaEventRecord(start));
  sgemm_2d_register_tiling<<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
  CHECK(cudaEventRecord(stop));

  CHECK(cudaGetLastError());
  CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK(cudaEventElapsedTime(&ms, start, stop));
  std::cout << "Kernel Execution Time (2D Register Tiling): " << ms << " ms\n";

  CHECK(cudaMemcpy(C_gpu.data(), dC, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
  bool pass = true;
  for (int i = 0; i < M * N; ++i) { if (!nearly_equal(C_cpu[i], C_gpu[i])) { pass = false; break; } }
  if (pass) std::cout << "Validation PASSED!" << std::endl;

  return 0;
}
