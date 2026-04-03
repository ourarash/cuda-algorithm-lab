/*
 * Siboehm GEMM: 1D Register Tiling
 *
 * Intention:
 * This version increases work per thread so each thread computes a small
 * vertical strip of output values in registers.
 *
 * High-Level Algorithm:
 * - Tile A and B through shared memory.
 * - Let each thread own TM outputs instead of just one.
 * - Keep those partial sums in registers across the full K-tile loop.
 * - Write the register results back at the end.
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
 * @brief Performs matrix multiplication (C = alpha * A * B + beta * C) using
 * shared memory and register blocking for improved performance.
 *
 * @param M, N, K Dimensions of the matrices (M-by-K, K-by-N -> M-by-N).
 * @param alpha, beta Scalar multipliers.
 * @param A, B, C Device pointers to the matrices.
 *
 * (Read here first!):
 * Main intuition: Compared to the shared memory version, this kernel coarsens
 * threads such that each thread computes TM (8) elements of the output matrix
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
__global__ void sgemm_register_blocked(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
  // --- Shared Memory Declaration ---
  // 1D arrays for flexible, coalesced loading.
  __shared__ float As[BM][BK]; // For the tile of A (64x8)
  __shared__ float Bs[BK][BN]; // For the tile of B (8x64)

  // --- Thread Indexing ---
  const int threadRow = threadIdx.y; // 0 to 7
  const int threadCol = threadIdx.x; // 0 to 63

  // --- Block Indexing ---
  // Identify the top-left corner of the C tile this block is responsible for.
  const int blockRow = blockIdx.y * BM;
  const int blockCol = blockIdx.x * BN;

  // --- Register-level Cache for Results ---
  // Each thread computes TM (8) output values.
  float threadResults[TM] = {0.0f};

  // --- Main Loop over K-dimension Tiles ---
  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {

    // --- Coalesced Loading from Global to Shared Memory ---
    // Each of the 512 threads in the block loads one element of As and one
    // element of Bs.
    const int threadId = threadIdx.y * blockDim.x + threadIdx.x;

    // Load a tile from A
    // Each thread loads one element of the tile of A
    int a_tile_row = threadId / BK;
    int a_tile_col = threadId % BK;
    int a_global_row = blockRow + a_tile_row;
    int a_global_col = bkIdx + a_tile_col;

    if (a_global_row < M && a_global_col < K) {
      As[a_tile_row][a_tile_col] = A[a_global_row * K + a_global_col];
    } else {
      As[a_tile_row][a_tile_col] = 0.0f;
    }

    // Load a tile from B
    // Each thread loads one element of the tile of B
    int b_tile_row = threadId / BN;
    int b_tile_col = threadId % BN;
    int b_global_row = bkIdx + b_tile_row;
    int b_global_col = blockCol + b_tile_col;

    if (b_global_row < K && b_global_col < N) {
      Bs[b_tile_row][b_tile_col] = B[b_global_row * N + b_global_col];
    } else {
      Bs[b_tile_row][b_tile_col] = 0.0f;
    }

    __syncthreads(); // Ensure tiles are fully loaded before computation

    // --- Tile-level Computation ---
    // The outer loop is over the K-dimension of the tile (dotIdx).
    // This allows the value from Bs to be reused TM times.
    // That loop is calculating the dot product of the tiles currently loaded
    // into shared memory.
    // (threadRow, threadCol) is responsible for computing TM elements in a
    // column of the final C tile.
    // Note that this is the same loop in the shared memory version, we wrote
    // like this before, but
    // for (int k = 0; k < BLOCK_SIZE; ++k)
    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // Load one value from Bs into a register
      // Btmp = Bs[dotIdx, threadCol]
      float Btmp = Bs[dotIdx][threadCol];

// Unroll the inner loop for performance.
#pragma unroll
      // Inner loop computes the TM results for this thread
      //
      // Moving in a column of As: For a fixed dotIdx, the resIdx loop
      // iterates from 0 to TM-1, effectively walking down the dotIdx-th
      // column of the A tile.
      // You’re accessing a column of the A tile (shared memory), from TM
      // different rows, one per resIdx.
      //
      // Building a partial sum for a column of C: Each of those TM values from
      // the As column is multiplied by the same Btmp value (which comes from a
      // row in the B tile). These products are then added to the corresponding
      // TM elements in the threadResults array, which holds the partial sums
      // for the final column of C that this thread is responsible for.

      for (int resIdx = 0; resIdx < TM; ++resIdx) {
        // Reuse Btmp TM times
        threadResults[resIdx] += As[threadRow * TM + resIdx][dotIdx] * Btmp;
      }
    }

    __syncthreads(); // Ensure all computations are done before loading the next
                     // tile
  }

// --- Write Results from Registers to Global Memory ---
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    int c_row = blockRow + threadRow * TM + i;
    int c_col = blockCol + threadCol;

    if (c_row < M && c_col < N) {
      // Apply alpha and beta scalars
      C[c_row * N + c_col] =
          alpha * threadResults[i] + beta * C[c_row * N + c_col];
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
  sgemm_register_blocked<<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta,
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
