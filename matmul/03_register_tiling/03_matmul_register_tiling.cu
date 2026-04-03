/*
 * Register-Tiled Matrix Multiplication
 *
 * Intention:
 * This file increases work per thread so each thread reuses values from shared
 * memory more aggressively and performs more math per memory fetch.
 *
 * High-Level Algorithm:
 * - Use shared-memory tiling for the block-level data movement.
 * - Give each thread responsibility for a vertical strip of output values.
 * - Keep those partial sums in registers.
 * - Reuse each loaded B value across several outputs computed by the same
 *   thread.
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
// These constants define the architecture of our matrix multiplication.

// The dimensions of a tile processed by a single thread block.
// We will process a 64x64 tile of C in each block.
constexpr int BM = 64; // Block size in M dimension
constexpr int BN = 64; // Block size in N dimension

// The "inner" dimension for the dot product loop.
// This is the size of the tile loaded into shared memory along the K-axis.
constexpr int BK = 8;

// Work per thread (Register-level tiling)
// Each thread will compute a TM x 1 column vector of C.
constexpr int TM = 8;

// Thread block dimensions
// The number of threads in a block.
constexpr int BLOCK_DIM_X = BN;      // 64 threads in X dimension
constexpr int BLOCK_DIM_Y = BM / TM; // 8 threads in Y dimension
// Total threads per block = 64 * 8 = 512

/**
 * 3. Work-per-thread via Register Tiling
 * This version increases arithmetic intensity by having each thread compute
 * more than one output element. Each thread computes an 8x1 column of the
 * output C-tile. It loads a value from shared memory into a private register
 * and reuses that value 8 times. This reduces traffic to shared memory and
 * increases the ratio of math-to-memory operations.
 */
__global__ void sgemm_register_tiling(int M, int N, int K, float alpha,
                                      const float *A, const float *B,
                                      float beta, float *C) {
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  const int threadRow = threadIdx.y; // 0 to 7
  const int threadCol = threadIdx.x; // 0 to 63

  const int blockRow = blockIdx.y * BM;
  const int blockCol = blockIdx.x * BN;

  float threadResults[TM] = {0.0f};

  // Loop over K dimension in chunks/tiles of size BK
  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    const int threadId = threadIdx.y * blockDim.x + threadIdx.x;

    // Load a tile of A from global memory into shared memory
    int a_tile_row = threadId / BK;
    int a_tile_col = threadId % BK;
    int a_global_row = blockRow + a_tile_row;
    int a_global_col = bkIdx + a_tile_col;

    if (a_global_row < M && a_global_col < K) {
      As[a_tile_row][a_tile_col] = A[a_global_row * K + a_global_col];
    } else {
      As[a_tile_row][a_tile_col] = 0.0f;
    }

    // Load a tile of B from global memory into shared memory
    int b_tile_row = threadId / BN;
    int b_tile_col = threadId % BN;
    int b_global_row = bkIdx + b_tile_row;
    int b_global_col = blockCol + b_tile_col;

    if (b_global_row < K && b_global_col < N) {
      Bs[b_tile_row][b_tile_col] = B[b_global_row * N + b_global_col];
    } else {
      Bs[b_tile_row][b_tile_col] = 0.0f;
    }

    // Synchronize to ensure all threads have finished loading the tiles
    __syncthreads();

    // Compute the dot product for the current tiles
    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      float regB = Bs[dotIdx][threadCol];
#pragma unroll
      // Accumulate partial sums into thread-local registers
      for (int resultIndex = 0; resultIndex < TM; ++resultIndex) {
        threadResults[resultIndex] += As[threadRow * TM + resultIndex][dotIdx] * regB;
      }
    }

    // Synchronize to ensure all threads have finished computing before overwriting shared memory
    __syncthreads();
  }

// Write the accumulated results from registers back to global memory C
#pragma unroll
  for (int resultIndex = 0; resultIndex < TM; ++resultIndex) {
    int c_row = blockRow + threadRow * TM + resultIndex;
    int c_col = blockCol + threadCol;

    if (c_row < M && c_col < N) {
      // C = α*(A@B)+β*C
      C[c_row * N + c_col] =
          alpha * threadResults[resultIndex] + beta * C[c_row * N + c_col];
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

  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1);
  dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y, 1);

  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));

  CHECK(cudaEventRecord(start));
  sgemm_register_tiling<<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta,
                                               dC);
  CHECK(cudaEventRecord(stop));

  CHECK(cudaGetLastError());
  CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK(cudaEventElapsedTime(&ms, start, stop));
  std::cout << "Kernel Execution Time (Register Tiling): " << ms << " ms\n";

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
