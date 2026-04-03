/*
 * Siboehm GEMM: Linearized Coalesced Mapping
 *
 * Intention:
 * This version shows that the same coalesced computation can be expressed with
 * a linear thread index rather than a 2D thread layout.
 *
 * High-Level Algorithm:
 * - Flatten the local thread id.
 * - Derive the output row and column from that single id.
 * - Preserve the coalesced access pattern from the 2D coalesced version.
 * - Compute one output value per thread.
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

constexpr int kBlockSize = 32;

// This kernel computes the same coalesced mapping as the 2D version, but it
// derives row/column coordinates from a flattened local thread id.
__global__ void sgemm_naive_coalesced_linear(int M, int N, int K, float alpha,
                                             const float *A, const float *B,
                                             float beta, float *C) {
  // Flatten the 2D thread coordinates into one local id in the 32x32 tile.
  const int linear_thread_id = threadIdx.y * blockDim.x + threadIdx.x;

  // Recover the output row/column from the flattened local id.
  const int i = blockIdx.y * kBlockSize + (linear_thread_id / kBlockSize);
  const int j = blockIdx.x * kBlockSize + (linear_thread_id % kBlockSize);

  // compute position in C that this thread is responsible for
  // Note that j changes faster than i inside the flattened thread order.
  // so access of
  // - A[i, k] is broadcast (since i, k are constant in the warp)
  // - B[k, j] is coalesced (since k, j are constant in the warp)
  // - C[i, j] is coalesced

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
  dim3 gridDim(CEIL_DIV(N, kBlockSize), CEIL_DIV(M, kBlockSize), 1);

  // kBlockSize * kBlockSize threads cooperate on one 32x32 output tile.
  dim3 blockDim(kBlockSize, kBlockSize, 1);
  // launch the asynchronous execution of the kernel on the device
  // The function call returns immediately on the host
  sgemm_naive_coalesced_linear<<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB,
                                                      beta, dC);
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
