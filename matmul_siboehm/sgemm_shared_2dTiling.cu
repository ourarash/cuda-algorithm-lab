/*
 * Siboehm GEMM: 2D Register Tiling
 *
 * Intention:
 * This version extends register tiling from a 1D strip to a 2D patch so each
 * thread produces a small tile of the final matrix.
 *
 * High-Level Algorithm:
 * - Stage block tiles of A and B in shared memory.
 * - Load a tiny TM x TN work tile into registers per thread.
 * - Accumulate that tile across the K dimension.
 * - Store the completed TM x TN patch back into C.
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
constexpr int TM = 4;
constexpr int TN = 4;

// Thread block dimensions
// The number of threads in a block.
constexpr int BLOCK_DIM_X = BN / TN; // 8 threads in X dimension
constexpr int BLOCK_DIM_Y = BM / TM; // 8 threads in Y dimension
// Total threads per block = 8 * 8 = 64

/**
 * @brief Performs matrix multiplication (C = alpha * A * B + beta * C) using
 * shared memory and register blocking for improved performance.
 *
 * @param M, N, K Dimensions of the matrices (M-by-K, K-by-N -> M-by-N).
 * @param alpha, beta Scalar multipliers.
 * @param A, B, C Device pointers to the matrices.
 *
 * (Read here first!):
 * Main intuition: Compared to 1d blocking, this kernel coarsens
 * threads such that each thread computes TMxTN elements of the output matrix
 * C, instead of just one.
 *
 * Kernel breakdown:
 * - Each thread block computes a BM-by-BN (64x64) tile of the output matrix C.
 * - Each thread within the block computes a TM-by-1 (8x1) column vector of that
 * C tile.
 * - The main loop iterates through the K dimension in chunks of size BK (8).
 * - In each iteration, the block loads a BM-by-BK tile of A and a BK-by-BN tile
 * of B into shared memory.
 * - A nested loop then computes the matrix multiplication for the tiles,
 * accumulating the results in registers (threadResults).
 */
__global__ void sgemm_register_blocked_2d(int M, int N, int K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C) {
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  // --- Thread Indexing ---
  const int threadRow = threadIdx.y; // 0 to blockDim.y-1
  const int threadCol = threadIdx.x; // 0 to blockDim.x-1

  // --- Block origin in C ---
  const int blockRow = blockIdx.y * BM;
  const int blockCol = blockIdx.x * BN;

  // --- Register tile for C: TM × TN elements per thread ---
  float threadResults[TM][TN] = {0.0f};

  // --- Loop over BK tiles along K dimension ---
  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    const int threadId = threadIdx.y * blockDim.x + threadIdx.x;

    // --- Load A tile into shared memory ---
    for (int row = threadId; row < BM * BK; row += blockDim.x * blockDim.y) {
      int i = row / BK;
      int j = row % BK;
      int global_i = blockRow + i;
      int global_j = bkIdx + j;
      As[i][j] =
          (global_i < M && global_j < K) ? A[global_i * K + global_j] : 0.0f;
    }

    // --- Load B tile into shared memory ---
    for (int row = threadId; row < BK * BN; row += blockDim.x * blockDim.y) {
      int i = row / BN;
      int j = row % BN;
      int global_i = bkIdx + i;
      int global_j = blockCol + j;
      Bs[i][j] =
          (global_i < K && global_j < N) ? B[global_i * N + global_j] : 0.0f;
    }

    __syncthreads();

    // --- Compute TM×TN outer product ---
    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      float regM[TM], regN[TN];

#pragma unroll
      for (int i = 0; i < TM; ++i)
        regM[i] = As[threadRow * TM + i][dotIdx];

#pragma unroll
      for (int j = 0; j < TN; ++j)
        regN[j] = Bs[dotIdx][threadCol * TN + j];

#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          threadResults[i][j] += regM[i] * regN[j];
    }

    __syncthreads();
  }

// --- Store results back to C ---
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    int c_row = blockRow + threadRow * TM + i;

#pragma unroll
    for (int j = 0; j < TN; ++j) {
      int c_col = blockCol + threadCol * TN + j;

      if (c_row < M && c_col < N) {
        C[c_row * N + c_col] =
            alpha * threadResults[i][j] + beta * C[c_row * N + c_col];
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
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "Max threads per SM: " << prop.maxThreadsPerMultiProcessor
            << "\n";

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

  // CPU reference
  cpu_gemm(M, N, K, alpha, A.data(), B.data(), beta, C_cpu.data());

  // Debug: print first few values
  std::cout << "First few CPU results: ";
  for (int i = 0; i < 5; ++i) {
    std::cout << C_cpu[i] << " ";
  }
  std::cout << std::endl;

  // Allocate GPU memory
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

  // create as many blocks as necessary to map all of C
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1);

  // Define the block dimensions based on the constants
  dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y, 1);

  // launch the asynchronous execution of the kernel on the device
  sgemm_register_blocked_2d<<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta,
                                                   dC);
  CHECK(cudaGetLastError()); // Check for kernel launch errors
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaMemcpy(C_gpu.data(), dC, C_gpu.size() * sizeof(float),
                   cudaMemcpyDeviceToHost));

  // Debug: print first few GPU results
  std::cout << "First few GPU results: ";
  for (int i = 0; i < 5; ++i) {
    std::cout << C_gpu[i] << " ";
  }
  std::cout << std::endl;

  // Compare
  for (int i = 0; i < M * N; ++i) {
    if (!nearly_equal(C_cpu[i], C_gpu[i])) {
      std::cerr << "Mismatch at " << i << ": CPU=" << C_cpu[i]
                << ", GPU=" << C_gpu[i] << std::endl;
      return 1;
    }
  }

  std::cout << "PASS: CPU and GPU results match." << std::endl;

  CHECK(cudaFree(dA));
  CHECK(cudaFree(dB));
  CHECK(cudaFree(dC));
  return 0;
}
