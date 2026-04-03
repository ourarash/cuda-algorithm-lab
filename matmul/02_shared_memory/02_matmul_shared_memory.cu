/*
 * Shared Memory Tiled Matrix Multiplication
 *
 * Intention:
 * This file introduces block tiling with shared memory to reduce redundant
 * global-memory traffic.
 *
 * High-Level Algorithm:
 * - Assign each thread block to one output tile of C.
 * - Cooperatively load one tile of A and one tile of B into shared memory.
 * - Reuse those tiles for many multiply-accumulate operations before loading
 *   the next K tile.
 * - Pad the B tile by one column to avoid shared-memory bank conflicts.
 */
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

#define TILE_SIZE 32 // Tile size for shared memory

/**
 * 2. Shared Memory Tiling Approach
 * This kernel uses Shared Memory to drastically reduce global memory accesses.
 * Each thread block is assigned to one tile of the output matrix C, and each
 * thread in that block computes one element within that C tile.
 * Threads within the block cooperatively load the corresponding tiles of A and
 * B into ultra-fast shared memory, reuse them for their dot product
 * calculations, and pad the B tile by 1 column to avoid shared memory bank
 * conflicts.
 */
__global__ void sgemm_shared(int M, int N, int K, float alpha, const float *A,
                             const float *B, float beta, float *C) {
  // Allocate shared memory for tiles of A and B.
  // Padding tileB by 1 column prevents shared memory bank conflicts when threads access it.
  __shared__ float tileA[TILE_SIZE][TILE_SIZE];
  __shared__ float tileB[TILE_SIZE][TILE_SIZE + 1];

  // Global row and column index in the output matrix C
  int globalRow = blockIdx.y * TILE_SIZE + threadIdx.y;
  int globalCol = blockIdx.x * TILE_SIZE + threadIdx.x;

  float partialSum = 0.0f;

  // Loop over the tiles of the shared K dimension.
  // This block owns one TILE_SIZE x TILE_SIZE output tile of C, but computing
  // that tile requires accumulating products across the full K dimension.
  // In each iteration, the block loads one tile of A and one tile of B into
  // shared memory, computes this tile's partial contribution to C, and then
  // moves to the next K tile.
  for (int tileIdx = 0; tileIdx < K / TILE_SIZE; tileIdx++) {
    // Each thread loads one element of the current A tile.
    int aRow = globalRow;
    int aCol = tileIdx * TILE_SIZE + threadIdx.x;

    // Each thread also loads one element of the current B tile.
    // Over the full loop, each thread loads multiple A/B elements, one pair
    // per tileIdx iteration.
    int bRow = tileIdx * TILE_SIZE + threadIdx.y;
    int bCol = globalCol;

    // Load elements into shared memory (with bounds checking)
    if (aRow < M && aCol < K) {
      tileA[threadIdx.y][threadIdx.x] = A[aRow * K + aCol];
    } else {
      tileA[threadIdx.y][threadIdx.x] = 0.0f;
    }

    if (bRow < K && bCol < N) {
      tileB[threadIdx.y][threadIdx.x] = B[bRow * N + bCol];
    } else {
      tileB[threadIdx.y][threadIdx.x] = 0.0f;
    }

    // Wait for all threads in the block to finish loading their elements into shared memory
    __syncthreads();

    // Multiply row of A with column of B
    for (int k = 0; k < TILE_SIZE; k++) {
      partialSum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
    }

    // Wait for all threads to finish computing before loading the next tile
    __syncthreads();
  }

  // Write the accumulated result to global memory, applying alpha and beta
  if (globalRow < M && globalCol < N) {
    // C = α*(A@B)+β*C
    C[globalRow * N + globalCol] =
        alpha * partialSum + beta * C[globalRow * N + globalCol];
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

  CHECK(cudaMemcpy(dA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dB, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dC, C_gpu.data(), C_gpu.size() * sizeof(float),
                   cudaMemcpyHostToDevice));

  dim3 blockSize(TILE_SIZE, TILE_SIZE);
  dim3 gridSize(CEIL_DIV(N, TILE_SIZE), CEIL_DIV(M, TILE_SIZE), 1);

  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));

  CHECK(cudaEventRecord(start));
  sgemm_shared<<<gridSize, blockSize>>>(M, N, K, alpha, dA, dB, beta, dC);
  CHECK(cudaEventRecord(stop));

  CHECK(cudaGetLastError());
  CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK(cudaEventElapsedTime(&ms, start, stop));
  std::cout << "Kernel Execution Time (Shared Memory): " << ms << " ms\n";

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
