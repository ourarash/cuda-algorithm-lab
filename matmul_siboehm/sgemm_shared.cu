#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define BLOCK_SIZE 32 // Tile size for shared memory

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

// Threads in warp change
__global__ void sgemm_naive_shared(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {
  // Declare shared memory tiles to load submatrices of A and B into faster
  // memory
  __shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];

  // Global row and column index in the output matrix C
  int globalRow = blockIdx.y * BLOCK_SIZE + threadIdx.y;
  int globalCol = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  float partialSum = 0.0f;

  // Loop over tiles along the shared dimension (K)
  for (int tileIdx = 0; tileIdx < (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
       tileIdx++) {
    // Compute global indices for the current tile of A
    int aRow = globalRow;
    int aCol = tileIdx * BLOCK_SIZE + threadIdx.x;

    // Compute global indices for the current tile of B
    int bRow = tileIdx * BLOCK_SIZE + threadIdx.y;
    int bCol = globalCol;

    // Load elements into shared memory with bounds check
    // Load tileA[row][col] = A[globalRow][aCol]
    tileA[threadIdx.y][threadIdx.x] =
        (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;

    // Load tileB[row][col] = B[bRow][globalCol]
    tileB[threadIdx.y][threadIdx.x] =
        (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;

    // Wait for all threads to load their tile before computation
    __syncthreads();

    // Multiply row of A with column of B (shared memory multiply-accumulate)
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      partialSum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
    }

    // Synchronize before loading the next tile
    __syncthreads();
  }

  // Store the result into C if within bounds
  // C[globalRow][globalCol] = α * (A @ B) + β * C
  if (globalRow < M && globalCol < N) {
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
  dim3 gridDim(CEIL_DIV(M, BLOCK_SIZE), CEIL_DIV(N, BLOCK_SIZE), 1);

  // BLOCK_SIZE * BLOCK_SIZE threads per block
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
  // launch the asynchronous execution of the kernel on the device
  // The function call returns immediately on the host
  sgemm_naive_shared<<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
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
