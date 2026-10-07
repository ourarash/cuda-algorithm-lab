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
 *
 * The host-side driver (inputs, CPU reference, validation, timing) lives in
 * ../gemm_harness.cuh and is shared by every stage.
 */
#include "../gemm_harness.cuh"

// --- Tiling and Block Dimensions ---
constexpr int BM = 128;  // Block size in M dimension
constexpr int BN = 128;  // Block size in N dimension
constexpr int BK = 8;    // Inner loop tile size

// Work per thread (2D Register-level tiling)
// Each thread will compute an 8x8 grid of C.
constexpr int TM = 8;
constexpr int TN = 8;

// Thread block dimensions
// Number of threads = (128/8) * (128/8) = 16 * 16 = 256
constexpr int BLOCK_DIM_X = BN / TN;
constexpr int BLOCK_DIM_Y = BM / TM;
constexpr int NUM_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y;

/**
 * 4. 2D Register Tiling
 * Building upon 1D register tiling, each thread now computes a 2D grid
 * (8x8) of output elements. It loads 8 elements from the A tile and 8
 * elements from the B tile into local registers, then performs 64
 * multiply-accumulate operations. That is 64 FMAs per 16 shared-memory loads,
 * compared with 8 FMAs per 9 loads in the 1D version.
 *
 * What still limits it (and what the next stage, 04_vectorized, changes):
 * - Every shared-memory load is a separate 32-bit instruction.
 * - The shared-memory reads have bank conflicts. A warp here is two rows of
 *   16 threads. Reading Bs[dotIdx][threadCol * TN + j], threads whose
 *   threadCol differs by 4 are 32 words apart, so they hit the same bank:
 *   a 4-way conflict. Reading As[threadRow * TM + i][dotIdx], the warp's two
 *   threadRows are 64 words apart: a 2-way conflict.
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
    for (int loadOffset = 0; loadOffset < BM * BK; loadOffset += NUM_THREADS) {
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
    for (int loadOffset = 0; loadOffset < BK * BN; loadOffset += NUM_THREADS) {
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
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        regM[i] = As[threadRow * TM + i][dotIdx];
      }
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        regN[j] = Bs[dotIdx][threadCol * TN + j];
      }

      // Perform 64 FMAs in registers for this K step.
      // This is the key advantage over 1D tiling: one set of loaded register
      // values updates an 8x8 patch instead of only a row or a column, so the
      // thread does more math per shared-memory read.
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
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

void launch_sgemm_2d_register_tiling(int M, int N, int K, float alpha,
                                     const float *A, const float *B,
                                     float beta, float *C) {
  dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
  dim3 grid(lab::ceil_div(N, BN), lab::ceil_div(M, BM));
  sgemm_2d_register_tiling<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

int main(int argc, char **argv) {
  return run_gemm<float>("2D register tiling", argc, argv, {1024, 1024, 1024},
                         {257, 129, 95}, launch_sgemm_2d_register_tiling);
}
