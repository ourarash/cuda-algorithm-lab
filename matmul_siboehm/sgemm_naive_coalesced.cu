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

// Threads in warp change
__global__ void sgemm_naive_coalesced(int M, int N, int K, float alpha,
                                      const float *A, const float *B,
                                      float beta, float *C) {
  // compute position in C that this thread is responsible for
  // Note that j is changing with threadIdx.x and i with threadIdx.y
  // This means j is changing faster than i
  // so access of
  // - A[i, k] is broadcased (since i, k are constant in the warp)
  // - B[k, j] is coalesced (since k, j are constant in the warp)
  // - C[i, j] is coalesced
  const uint j = blockIdx.x * blockDim.x + threadIdx.x;
  const uint i = blockIdx.y * blockDim.y + threadIdx.y;

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (i < M && j < N) {
    float tmp = 0.0;
    for (int k = 0; k < K; ++k) {
      tmp += A[i * K + k] * B[k * N + j]; // A[i, k] * B[k, j]
    }
    // C = α*(A@B)+β*C
    // C[i, j] = α * tmp + β * C[i, j];
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
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "Max threads per SM: " << prop.maxThreadsPerMultiProcessor
            << "\n";

  const int M = 1024, N = 1024, K = 1024;
  const int BLOCK_SIZE = 32; // Block size for the kernel
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
  sgemm_naive_coalesced<<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta,
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
