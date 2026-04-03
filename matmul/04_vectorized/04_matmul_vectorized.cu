/*
 * Vectorized Matrix Multiplication
 *
 * Intention:
 * This file widens memory accesses with float4 loads and stores to improve
 * memory throughput compared with the scalar shared-memory version.
 *
 * High-Level Algorithm:
 * - Keep the tiled shared-memory structure from earlier GEMM kernels.
 * - Load B in float4 chunks so each thread fetches four neighboring values at
 *   once.
 * - Accumulate four outputs per thread in registers.
 * - Store the result back with vector-friendly access patterns.
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
 * 5. Vectorized Memory Access Approach (`float4`)
 * This kernel improves memory throughput by fetching 128-bit blocks (four 32-bit
 * floats) from global memory at once using the `float4` built-in type. 
 * It also computes 4 output values per thread (instruction-level parallelism) 
 * and stores them back using vectorized stores.
 */
__global__ void sgemm_vectorized(int M, int N, int K, float alpha,
                                 const float *A, const float *B, float beta,
                                 float *C) {
  // Shared memory for tiles. tileB is cast to float4, so its inner dimension is divided by 4.
  __shared__ float tileA[TILE_SIZE][TILE_SIZE];
  __shared__ float4 tileB[TILE_SIZE][TILE_SIZE / 4];

  // Determine the row and column in the global matrix C. 
  // threadIdx.x is multiplied by 4 because each thread processes 4 elements (a float4).
  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x * 4;

  // Accumulator for 4 output elements, stored in thread-local registers
  float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int tileIdx = 0; tileIdx < K / TILE_SIZE; ++tileIdx) {
    int aRow = row;
    int aCol = tileIdx * TILE_SIZE + threadIdx.x;
    int bRow = tileIdx * TILE_SIZE + threadIdx.y;
    int bCol = col;

    // Load 1 float from A into shared memory
    if (aRow < M && aCol < K)
      tileA[threadIdx.y][threadIdx.x] = A[aRow * K + aCol];
    else
      tileA[threadIdx.y][threadIdx.x] = 0.0f;

    // Load 4 floats (one float4) from B into shared memory using vectorized load
    if (bRow < K && bCol + 3 < N)
      tileB[threadIdx.y][threadIdx.x] =
          *(reinterpret_cast<const float4 *>(&B[bRow * N + bCol]));
    else
      tileB[threadIdx.y][threadIdx.x] = make_float4(0, 0, 0, 0);

    __syncthreads();

    // Compute dot product for the current tile
    for (int k = 0; k < TILE_SIZE; ++k) {
      float aVal = tileA[threadIdx.y][k];
      float4 bVal = tileB[k][threadIdx.x];
      // Broadcast aVal to all 4 elements of bVal
      acc.x += aVal * bVal.x;
      acc.y += aVal * bVal.y;
      acc.z += aVal * bVal.z;
      acc.w += aVal * bVal.w;
    }

    __syncthreads();
  }

  if (row < M && col + 3 < N) {
    // Read the existing 4 elements of C using a vectorized load
    float4 existing_C =
        *(reinterpret_cast<const float4 *>(&C[row * N + col]));
        
    // C = α*(A@B)+β*C applied to all 4 elements
    acc.x = alpha * acc.x + beta * existing_C.x;
    acc.y = alpha * acc.y + beta * existing_C.y;
    acc.z = alpha * acc.z + beta * existing_C.z;
    acc.w = alpha * acc.w + beta * existing_C.w;
    
    // Store the 4 updated elements back to C using a vectorized store
    *(reinterpret_cast<float4 *>(&C[row * N + col])) = acc;
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
  // Grid is sized for float4 operations
  dim3 gridSize(CEIL_DIV(N / 4, TILE_SIZE), CEIL_DIV(M, TILE_SIZE), 1);

  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));

  CHECK(cudaEventRecord(start));
  sgemm_vectorized<<<gridSize, blockSize>>>(M, N, K, alpha, dA, dB, beta, dC);
  CHECK(cudaEventRecord(stop));

  CHECK(cudaGetLastError());
  CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK(cudaEventElapsedTime(&ms, start, stop));
  std::cout << "Kernel Execution Time (Vectorized): " << ms << " ms\n";

  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));

  CHECK(cudaMemcpy(C_gpu.data(), dC, C_gpu.size() * sizeof(float),
                   cudaMemcpyDeviceToHost));

  bool pass = true;
  for (int i = 0; i < M * N; ++i) {
    if (!nearly_equal(C_cpu[i], C_gpu[i], 1e-3f)) { // Higher tolerance for FP32
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
