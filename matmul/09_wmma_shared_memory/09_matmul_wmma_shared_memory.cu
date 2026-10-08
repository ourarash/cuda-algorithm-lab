/*
 * Tensor Core Matrix Multiplication with Shared-Memory Staging (WMMA)
 *
 * Intention:
 * Stage 08 introduced the WMMA API but loaded every 16 x 16 fragment straight
 * from global memory, so each element of A and B was re-read by many warps.
 * This stage applies the lesson of stage 02 to Tensor Cores: the block first
 * stages a large tile of A and B in shared memory, and its warps then build
 * all of their fragments from that shared copy.
 *
 * High-Level Algorithm:
 * - Block tile 128 x 128 with BK = 32; 8 warps arranged 4 (M) x 2 (N).
 * - Warp tile 32 x 64 = 2 x 4 WMMA fragments of 16 x 16, so each warp keeps
 *   8 FP32 accumulator fragments.
 * - Per K tile: all 256 threads copy the A (128 x 32) and B (32 x 128) tiles
 *   to shared memory with 16-byte vector loads; __syncthreads(); each warp
 *   loads its fragments from shared memory and issues 2 x 4 x 2 mma_sync
 *   calls; __syncthreads().
 * - Every A fragment a warp loads is reused for 4 MMAs and every B fragment
 *   for 2, and every element loaded from global memory is used by the 2 or 4
 *   warps that share its rows or columns.
 *
 * Shared-memory padding: rows are padded by 8 halves (16 bytes). WMMA needs
 * the leading dimension to be a multiple of 8 halves and every fragment
 * pointer to be 32-byte aligned; the padding keeps both true while shifting
 * successive rows to different banks.
 *
 * Precision: FP16 inputs, FP32 accumulation (see stage 08).
 * Requirements: M, N, and K must be multiples of 16; compute capability 7.0+.
 *
 * The host-side driver (inputs, CPU reference, validation, timing, cuBLAS
 * baseline) lives in ../gemm_harness.cuh and is shared by every stage.
 */
#include <mma.h>

#include "../gemm_harness.cuh"

using namespace nvcuda;

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int WARPS_M = 4;
constexpr int WARPS_N = 2;
constexpr int NUM_THREADS = WARPS_M * WARPS_N * 32;  // 256
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int WARP_TILE_M = BM / WARPS_M;          // 32
constexpr int WARP_TILE_N = BN / WARPS_N;          // 64
constexpr int FRAGS_M = WARP_TILE_M / WMMA_M;      // 2
constexpr int FRAGS_N = WARP_TILE_N / WMMA_N;      // 4
constexpr int PAD = 8;                             // halves (16 bytes)
constexpr int LDA = BK + PAD;                      // 40
constexpr int LDB = BN + PAD;                      // 136
constexpr int VEC = 8;                             // halves per 16-byte load

static_assert(LDA % 8 == 0 && LDB % 8 == 0, "WMMA leading dimension");
static_assert((BM * BK / VEC) % NUM_THREADS == 0, "A tile loads evenly");
static_assert((BK * BN / VEC) % NUM_THREADS == 0, "B tile loads evenly");

/**
 * 9. WMMA with shared-memory staging
 */
__global__ void hgemm_wmma_shared(int M, int N, int K, float alpha,
                                  const half *A, const half *B, float beta,
                                  float *C) {
  __shared__ __align__(128) half As[BM][LDA];
  __shared__ __align__(128) half Bs[BK][LDB];

  const int blockRow = blockIdx.y * BM;
  const int blockCol = blockIdx.x * BN;
  const int warpIdx = threadIdx.x / 32;
  const int warpRow = warpIdx / WARPS_N;
  const int warpCol = warpIdx % WARPS_N;

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
      acc[FRAGS_M][FRAGS_N];
#pragma unroll
  for (int i = 0; i < FRAGS_M; ++i) {
#pragma unroll
    for (int j = 0; j < FRAGS_N; ++j) {
      wmma::fill_fragment(acc[i][j], 0.0f);
    }
  }

  for (int k0 = 0; k0 < K; k0 += BK) {
    // ---- Global -> shared, 8 halves (16 bytes) per load ----
    // K and N are multiples of 16, so a vector is either entirely inside the
    // matrix or entirely outside it.
    for (int idx = threadIdx.x; idx < BM * BK / VEC; idx += NUM_THREADS) {
      const int r = idx / (BK / VEC);
      const int c = (idx % (BK / VEC)) * VEC;
      const int row = blockRow + r;
      const int col = k0 + c;
      uint4 v = make_uint4(0, 0, 0, 0);  // Eight zero halves
      if (row < M && col < K) {
        v = *reinterpret_cast<const uint4 *>(&A[static_cast<size_t>(row) * K + col]);
      }
      *reinterpret_cast<uint4 *>(&As[r][c]) = v;
    }
    for (int idx = threadIdx.x; idx < BK * BN / VEC; idx += NUM_THREADS) {
      const int r = idx / (BN / VEC);
      const int c = (idx % (BN / VEC)) * VEC;
      const int row = k0 + r;
      const int col = blockCol + c;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (row < K && col < N) {
        v = *reinterpret_cast<const uint4 *>(&B[static_cast<size_t>(row) * N + col]);
      }
      *reinterpret_cast<uint4 *>(&Bs[r][c]) = v;
    }
    __syncthreads();

    // ---- Shared -> fragments -> Tensor Cores ----
#pragma unroll
    for (int kk = 0; kk < BK; kk += WMMA_K) {
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
          a_frag[FRAGS_M];
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
          b_frag[FRAGS_N];
#pragma unroll
      for (int i = 0; i < FRAGS_M; ++i) {
        wmma::load_matrix_sync(a_frag[i],
                               &As[warpRow * WARP_TILE_M + i * WMMA_M][kk], LDA);
      }
#pragma unroll
      for (int j = 0; j < FRAGS_N; ++j) {
        wmma::load_matrix_sync(b_frag[j],
                               &Bs[kk][warpCol * WARP_TILE_N + j * WMMA_N], LDB);
      }
#pragma unroll
      for (int i = 0; i < FRAGS_M; ++i) {
#pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) {
          wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
      }
    }
    __syncthreads();
  }

  // ---- Epilogue: C = alpha * acc + beta * C, one fragment at a time ----
  // M and N are multiples of 16, so a fragment is entirely inside C or
  // entirely outside it.
#pragma unroll
  for (int i = 0; i < FRAGS_M; ++i) {
#pragma unroll
    for (int j = 0; j < FRAGS_N; ++j) {
      const int row = blockRow + warpRow * WARP_TILE_M + i * WMMA_M;
      const int col = blockCol + warpCol * WARP_TILE_N + j * WMMA_N;
      if (row < M && col < N) {
        float *c_ptr = C + static_cast<size_t>(row) * N + col;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
        wmma::load_matrix_sync(c_frag, c_ptr, N, wmma::mem_row_major);
        for (int e = 0; e < c_frag.num_elements; ++e) {
          c_frag.x[e] = alpha * acc[i][j].x[e] + beta * c_frag.x[e];
        }
        wmma::store_matrix_sync(c_ptr, c_frag, N, wmma::mem_row_major);
      }
    }
  }
}

void launch_hgemm_wmma_shared(int M, int N, int K, float alpha, const half *A,
                              const half *B, float beta, float *C) {
  dim3 grid(lab::ceil_div(N, BN), lab::ceil_div(M, BM));
  hgemm_wmma_shared<<<grid, NUM_THREADS>>>(M, N, K, alpha, A, B, beta, C);
}

int main(int argc, char **argv) {
  GemmRequirements req;
  req.m_multiple = 16;
  req.n_multiple = 16;
  req.k_multiple = 16;
  req.min_compute_capability = 70;
  return run_gemm<half>("WMMA + shared memory", argc, argv, {1024, 1024, 1024},
                        {144, 80, 48}, launch_hgemm_wmma_shared, req);
}
