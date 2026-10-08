/*
 * Double-Buffered Matrix Multiplication with cp.async
 *
 * Intention:
 * Every previous stage alternates between two phases: all threads load a K
 * tile into shared memory, wait at a barrier, compute, wait again. While the
 * loads are in flight the math units idle, and while the math runs no loads
 * are in flight. This stage overlaps the two: it computes on one shared-memory
 * buffer while the next K tile streams into a second buffer.
 *
 * High-Level Algorithm:
 * - Same warptiling as 06_warptiling (block 128 x 128, warp 64 x 64, thread
 *   4 x (8 x 4)), but with two copies of the shared tiles: As[2], Bs[2].
 * - Loads use cp.async (Ampere and newer): an asynchronous global -> shared
 *   copy that does not pass through registers. A thread issues the copy and
 *   continues; it waits for completion only when it needs the data.
 *     prologue:   issue tile 0 into buffer 0
 *     iteration t: issue tile t+1 into buffer (t+1) % 2
 *                  wait until tile t has arrived; __syncthreads()
 *                  compute on buffer t % 2; __syncthreads()
 * - The trailing __syncthreads() matters: iteration t+1 overwrites the
 *   buffer that iteration t just computed on, so every warp must be done
 *   reading it first.
 *
 * cp.async API:
 * __pipeline_memcpy_async / __pipeline_commit / __pipeline_wait_prior from
 * <cuda_pipeline.h>. On compute capability 8.0+ they compile to the cp.async
 * instructions; on older GPUs they fall back to ordinary synchronous copies,
 * so this file builds and runs everywhere (without the overlap). Copies are
 * 4, 8, or 16 bytes; the `zfill` argument zero-fills the tail of a copy, which
 * handles tiles that hang over the matrix edge without a separate branch.
 * B is copied 16 bytes at a time. A is stored transposed in shared memory,
 * and a transposing copy cannot be one contiguous 16-byte block, so A uses
 * 4-byte copies, and consecutive threads write addresses BM floats apart,
 * which causes shared-memory bank conflicts on those writes. Stage 10 avoids
 * the transpose entirely by loading operands with ldmatrix.
 *
 * Requirements: N and K must be multiples of 4.
 *
 * The host-side driver (inputs, CPU reference, validation, timing, cuBLAS
 * baseline) lives in ../gemm_harness.cuh and is shared by every stage.
 */
#include <cuda_pipeline.h>

#include "../gemm_harness.cuh"

constexpr int NUM_THREADS = 128;
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 16;
constexpr int WM = 64;
constexpr int WN = 64;
constexpr int WNITER = 4;
constexpr int TM = 8;
constexpr int TN = 4;
constexpr int WARP_SIZE = 32;
constexpr int STAGES = 2;

constexpr int WMITER = (WM * WN) / (WARP_SIZE * TM * TN * WNITER);  // 1
constexpr int WSUBM = WM / WMITER;                                  // 64
constexpr int WSUBN = WN / WNITER;                                  // 16

static_assert((BM / WM) * (BN / WN) * WARP_SIZE == NUM_THREADS,
              "one warp per warp tile");
static_assert((WSUBM / TM) * (WSUBN / TN) == WARP_SIZE,
              "a warp's threads exactly cover one warp subtile");
static_assert((BM * BK) % NUM_THREADS == 0, "A tile loads evenly");
static_assert((BK * BN) % (4 * NUM_THREADS) == 0, "B tile loads evenly");

__device__ __forceinline__ float4 load_float4(const float *p) {
  return *reinterpret_cast<const float4 *>(p);
}

// Issues the asynchronous copies of one K tile into stage `s` of the shared
// buffers. Elements outside the matrices are zero-filled.
__device__ __forceinline__ void load_tile_async(
    float (*As)[BK][BM], float (*Bs)[BK][BN], int s, int k0, int M, int N,
    int K, const float *A, const float *B, int blockRow, int blockCol) {
  // A: one float per copy, written transposed. Consecutive threads read
  // consecutive K values of one row, so the global reads stay coalesced.
  for (int idx = threadIdx.x; idx < BM * BK; idx += NUM_THREADS) {
    const int m = idx / BK;
    const int k = idx % BK;
    const int row = blockRow + m;
    const int col = k0 + k;
    const bool valid = row < M && col < K;
    // An out-of-range copy reads nothing (zfill = whole size), but it still
    // needs a valid global address, so point it at A itself.
    const float *src = valid ? &A[static_cast<size_t>(row) * K + col] : A;
    __pipeline_memcpy_async(&As[s][k][m], src, sizeof(float),
                            valid ? 0 : sizeof(float));
  }
  // B: 16 bytes (one float4) per copy.
  for (int idx = threadIdx.x; idx < BK * BN / 4; idx += NUM_THREADS) {
    const int k = idx / (BN / 4);
    const int n = (idx % (BN / 4)) * 4;
    const int row = k0 + k;
    const int col = blockCol + n;
    const bool valid = row < K && col < N;  // N % 4 == 0
    const float *src = valid ? &B[static_cast<size_t>(row) * N + col] : B;
    __pipeline_memcpy_async(&Bs[s][k][n], src, 4 * sizeof(float),
                            valid ? 0 : 4 * sizeof(float));
  }
  __pipeline_commit();  // Close this tile's group of copies.
}

/**
 * 7. Double buffering with cp.async
 */
__global__ void sgemm_double_buffering(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
  __shared__ __align__(16) float As[STAGES][BK][BM];  // Transposed: [k][m]
  __shared__ __align__(16) float Bs[STAGES][BK][BN];

  const int blockRow = blockIdx.y * BM;
  const int blockCol = blockIdx.x * BN;

  const int warpIdx = threadIdx.x / WARP_SIZE;
  const int warpRow = warpIdx / (BN / WN);
  const int warpCol = warpIdx % (BN / WN);
  const int lane = threadIdx.x % WARP_SIZE;
  const int threadRowInWarp = lane / (WSUBN / TN);
  const int threadColInWarp = lane % (WSUBN / TN);

  float acc[WMITER * TM][WNITER * TN] = {};
  float regM[WMITER * TM];
  float regN[WNITER * TN];

  const int numTiles = lab::ceil_div(K, BK);

  // Prologue: start fetching the first tile.
  load_tile_async(As, Bs, 0, 0, M, N, K, A, B, blockRow, blockCol);

  for (int t = 0; t < numTiles; ++t) {
    const int cur = t % STAGES;

    if (t + 1 < numTiles) {
      // Start fetching the next tile into the other buffer, then wait until
      // at most one group (that next tile) is still in flight, which means
      // tile t has landed.
      load_tile_async(As, Bs, (t + 1) % STAGES, (t + 1) * BK, M, N, K, A, B,
                      blockRow, blockCol);
      __pipeline_wait_prior(1);
    } else {
      __pipeline_wait_prior(0);  // Last tile: wait for everything.
    }
    // Each thread waited only for its own copies; the barrier makes every
    // thread's copies visible to the whole block.
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; ++k) {
#pragma unroll
      for (int wSubRow = 0; wSubRow < WMITER; ++wSubRow) {
#pragma unroll
        for (int i = 0; i < TM; i += 4) {
          const float4 v = load_float4(
              &As[cur][k][warpRow * WM + wSubRow * WSUBM + threadRowInWarp * TM + i]);
          regM[wSubRow * TM + i + 0] = v.x;
          regM[wSubRow * TM + i + 1] = v.y;
          regM[wSubRow * TM + i + 2] = v.z;
          regM[wSubRow * TM + i + 3] = v.w;
        }
      }
#pragma unroll
      for (int wSubCol = 0; wSubCol < WNITER; ++wSubCol) {
#pragma unroll
        for (int j = 0; j < TN; j += 4) {
          const float4 v = load_float4(
              &Bs[cur][k][warpCol * WN + wSubCol * WSUBN + threadColInWarp * TN + j]);
          regN[wSubCol * TN + j + 0] = v.x;
          regN[wSubCol * TN + j + 1] = v.y;
          regN[wSubCol * TN + j + 2] = v.z;
          regN[wSubCol * TN + j + 3] = v.w;
        }
      }
#pragma unroll
      for (int i = 0; i < WMITER * TM; ++i) {
#pragma unroll
        for (int j = 0; j < WNITER * TN; ++j) {
          acc[i][j] += regM[i] * regN[j];
        }
      }
    }

    // Everyone must finish reading buffer `cur` before the next iteration
    // starts overwriting it with tile t + 2.
    __syncthreads();
  }

  // ---- Epilogue: C = alpha * acc + beta * C, float4 at a time ----
#pragma unroll
  for (int wSubRow = 0; wSubRow < WMITER; ++wSubRow) {
#pragma unroll
    for (int wSubCol = 0; wSubCol < WNITER; ++wSubCol) {
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        const int row = blockRow + warpRow * WM + wSubRow * WSUBM +
                        threadRowInWarp * TM + i;
#pragma unroll
        for (int j = 0; j < TN; j += 4) {
          const int col = blockCol + warpCol * WN + wSubCol * WSUBN +
                          threadColInWarp * TN + j;
          if (row < M && col < N) {
            float4 *c_ptr = reinterpret_cast<float4 *>(
                &C[static_cast<size_t>(row) * N + col]);
            const int ai = wSubRow * TM + i;
            const int aj = wSubCol * TN + j;
            float4 c = *c_ptr;
            c.x = alpha * acc[ai][aj + 0] + beta * c.x;
            c.y = alpha * acc[ai][aj + 1] + beta * c.y;
            c.z = alpha * acc[ai][aj + 2] + beta * c.z;
            c.w = alpha * acc[ai][aj + 3] + beta * c.w;
            *c_ptr = c;
          }
        }
      }
    }
  }
}

void launch_sgemm_double_buffering(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {
  dim3 grid(lab::ceil_div(N, BN), lab::ceil_div(M, BM));
  sgemm_double_buffering<<<grid, NUM_THREADS>>>(M, N, K, alpha, A, B, beta, C);
}

int main(int argc, char **argv) {
  GemmRequirements req;
  req.n_multiple = 4;
  req.k_multiple = 4;
  return run_gemm<float>("Double buffering (cp.async)", argc, argv,
                         {1024, 1024, 1024}, {257, 132, 100},
                         launch_sgemm_double_buffering, req);
}
