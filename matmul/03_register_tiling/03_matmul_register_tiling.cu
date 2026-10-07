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
 *
 * The host-side driver (inputs, CPU reference, validation, timing) lives in
 * ../gemm_harness.cuh and is shared by every stage.
 */
#include "../gemm_harness.cuh"

// --- Tiling and Block Dimensions ---
// These constants define the architecture of our matrix multiplication.

// The dimensions of a tile processed by a single thread block.
// We will process a 64x64 tile of C in each block.
constexpr int BM = 64;  // Block size in M dimension
constexpr int BN = 64;  // Block size in N dimension

// The "inner" dimension for the dot product loop.
// This is the size of the tile loaded into shared memory along the K-axis.
constexpr int BK = 8;

// Work per thread (Register-level tiling)
// Each thread will compute a TM x 1 column vector of C.
constexpr int TM = 8;

// Thread block dimensions
// The number of threads in a block.
constexpr int BLOCK_DIM_X = BN;       // 64 threads in X dimension
constexpr int BLOCK_DIM_Y = BM / TM;  // 8 threads in Y dimension
// Total threads per block = 64 * 8 = 512. The tile loads below rely on this:
// 512 threads load the 64x8 A tile and the 8x64 B tile with one element each.
static_assert(BLOCK_DIM_X * BLOCK_DIM_Y == BM * BK, "one A element per thread");
static_assert(BLOCK_DIM_X * BLOCK_DIM_Y == BK * BN, "one B element per thread");

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

  const int threadRow = threadIdx.y;  // 0 to 7
  const int threadCol = threadIdx.x;  // 0 to 63

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
      // Accumulate partial sums into thread-local registers
#pragma unroll
      for (int resultIndex = 0; resultIndex < TM; ++resultIndex) {
        threadResults[resultIndex] +=
            As[threadRow * TM + resultIndex][dotIdx] * regB;
      }
    }

    // Synchronize to ensure all threads have finished computing before
    // overwriting shared memory
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

void launch_sgemm_register_tiling(int M, int N, int K, float alpha,
                                  const float *A, const float *B, float beta,
                                  float *C) {
  dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
  dim3 grid(lab::ceil_div(N, BN), lab::ceil_div(M, BM));
  sgemm_register_tiling<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

int main(int argc, char **argv) {
  return run_gemm<float>("1D register tiling", argc, argv, {1024, 1024, 1024},
                         {257, 129, 95}, launch_sgemm_register_tiling);
}
