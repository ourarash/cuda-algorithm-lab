/*
 * Tensor Core Matrix Multiplication with mma.sync, ldmatrix, and cp.async
 *
 * Intention:
 * WMMA hides how fragments are laid out in registers. This stage drops one
 * level lower, to the PTX instructions that WMMA and libraries like CUTLASS
 * are built from, which gives full control over data movement:
 * - mma.sync.m16n8k16: one warp multiplies a 16 x 16 tile of A by a 16 x 8
 *   tile of B into a 16 x 8 FP32 tile, with a documented register layout.
 * - ldmatrix: one warp loads four 8 x 8 matrices of 16-bit values from shared
 *   memory straight into that register layout (.trans transposes on the
 *   fly, which is how the row-major B tile becomes the column operand).
 * - cp.async: asynchronous global -> shared copies, double-buffered as in
 *   stage 07 but written as raw PTX here.
 *
 * High-Level Algorithm:
 * - Block tile 128 x 128, BK = 32, two shared-memory stages.
 * - 8 warps arranged 2 (M) x 4 (N); warp tile 64 x 32 = 4 x 4 MMA tiles of
 *   16 x 8, so each thread holds 4 x 4 x 4 = 64 FP32 accumulators.
 * - Per K tile and per 16-wide K step, each warp issues 4 ldmatrix.x4 for A,
 *   2 ldmatrix.x4.trans for B, then 16 mma.sync.
 *
 * Swizzled shared memory (instead of padding):
 * ldmatrix reads 8 rows of 16 bytes for each 8 x 8 matrix. Without care those
 * 8 rows hit the same banks: an A tile row is 64 bytes, so rows r and r + 2
 * start in the same banks, and a B tile row is 256 bytes, so all 8 rows do.
 * The tiles are stored with the 16-byte chunks of each row permuted by XOR:
 *   A: physical chunk = chunk ^ ((row >> 1) & 3)   (4 chunks per row)
 *   B: physical chunk = chunk ^ (row & 7)           (16 chunks per row)
 * which sends the 8 rows of every ldmatrix read to 8 different groups of 4
 * banks. Unlike padding, swizzling wastes no shared memory and keeps every
 * 16-byte chunk aligned for cp.async.
 *
 * Fragment layouts (PTX ISA, "Matrix Fragments for mma.m16n8k16"), with
 * g = lane / 4 and t = lane % 4:
 *   A (16 x 16, row-major): {a0,a1} = A[g][2t..2t+1], {a2,a3} = A[g+8][2t..],
 *                           {a4,a5} = A[g][2t+8..],   {a6,a7} = A[g+8][2t+8..]
 *   B (16 x 8, column):     {b0,b1} = B[2t..2t+1][g], {b2,b3} = B[2t+8..][g]
 *   C (16 x 8, FP32):       c0,c1 = C[g][2t], C[g][2t+1]; c2,c3 = row g+8
 * Each 32-bit register holds two halves.
 *
 * Requirements: compute capability 8.0+ (mma.m16n8k16 with FP16 and cp.async);
 * N and K must be multiples of 8 so every 16-byte chunk is entirely inside or
 * outside the matrix. M can be anything.
 *
 * The host-side driver (inputs, CPU reference, validation, timing, cuBLAS
 * baseline) lives in ../gemm_harness.cuh and is shared by every stage.
 */
#include <cstdint>

#include "../gemm_harness.cuh"

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int STAGES = 2;
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int NUM_THREADS = WARPS_M * WARPS_N * 32;  // 256
constexpr int WARP_TILE_M = BM / WARPS_M;            // 64
constexpr int WARP_TILE_N = BN / WARPS_N;            // 32
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;
constexpr int MI = WARP_TILE_M / MMA_M;  // 4 MMA tiles along M per warp
constexpr int NJ = WARP_TILE_N / MMA_N;  // 4 MMA tiles along N per warp
constexpr int CHUNK = 8;                 // halves per 16-byte chunk
constexpr int A_CHUNKS_PER_ROW = BK / CHUNK;  // 4
constexpr int B_CHUNKS_PER_ROW = BN / CHUNK;  // 16

static_assert(MI * MMA_M == WARP_TILE_M && NJ * MMA_N == WARP_TILE_N,
              "MMA tiles exactly cover the warp tile");
static_assert(BK % MMA_K == 0 && STAGES == 2, "K tile and double buffering");
static_assert(A_CHUNKS_PER_ROW == 4, "A swizzle below assumes 4 chunks/row");
static_assert(B_CHUNKS_PER_ROW % 8 == 0, "B swizzle below permutes 8 chunks");
static_assert(NJ % 2 == 0, "B is loaded two n8 tiles per ldmatrix.x4");

// Offsets (in halves) of chunk `chunk` of row `row` in the swizzled tiles.
__device__ __forceinline__ int a_offset(int row, int chunk) {
  return row * BK + (chunk ^ ((row >> 1) & 3)) * CHUNK;
}
__device__ __forceinline__ int b_offset(int row, int chunk) {
  return row * BN + (chunk ^ (row & 7)) * CHUNK;
}

__device__ __forceinline__ uint32_t smem_addr(const void *p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// The PTX below needs compute capability 8.0. The bodies are compiled out for
// older targets; the host never launches this kernel on those GPUs.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#define LAB_SM80_PTX 0
#else
#define LAB_SM80_PTX 1
#endif

// 16-byte asynchronous copy; src_bytes = 0 zero-fills without reading.
__device__ __forceinline__ void cp_async_16(uint32_t dst, const void *src,
                                            int src_bytes) {
#if LAB_SM80_PTX
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst),
               "l"(src), "r"(src_bytes));
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if LAB_SM80_PTX
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
#if LAB_SM80_PTX
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t addr, uint32_t &r0,
                                            uint32_t &r1, uint32_t &r2,
                                            uint32_t &r3) {
#if LAB_SM80_PTX
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
#endif
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t addr, uint32_t &r0,
                                                  uint32_t &r1, uint32_t &r2,
                                                  uint32_t &r3) {
#if LAB_SM80_PTX
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(addr));
#endif
}

// d += a * b for one 16 x 8 x 16 tile.
__device__ __forceinline__ void mma_16816(float (&d)[4], const uint32_t (&a)[4],
                                          const uint32_t (&b)[2]) {
#if LAB_SM80_PTX
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

// Issues the cp.async copies of one K tile into stage `s`.
__device__ __forceinline__ void load_tile(half *As, half *Bs, int k0, int M,
                                          int N, int K, const half *A,
                                          const half *B, int blockRow,
                                          int blockCol) {
  for (int idx = threadIdx.x; idx < BM * A_CHUNKS_PER_ROW; idx += NUM_THREADS) {
    const int r = idx / A_CHUNKS_PER_ROW;
    const int c = idx % A_CHUNKS_PER_ROW;
    const int row = blockRow + r;
    const int col = k0 + c * CHUNK;
    const bool valid = row < M && col < K;
    const half *src = valid ? &A[static_cast<size_t>(row) * K + col] : A;
    cp_async_16(smem_addr(&As[a_offset(r, c)]), src, valid ? 16 : 0);
  }
  for (int idx = threadIdx.x; idx < BK * B_CHUNKS_PER_ROW; idx += NUM_THREADS) {
    const int r = idx / B_CHUNKS_PER_ROW;
    const int c = idx % B_CHUNKS_PER_ROW;
    const int row = k0 + r;
    const int col = blockCol + c * CHUNK;
    const bool valid = row < K && col < N;
    const half *src = valid ? &B[static_cast<size_t>(row) * N + col] : B;
    cp_async_16(smem_addr(&Bs[b_offset(r, c)]), src, valid ? 16 : 0);
  }
  cp_async_commit();
}

/**
 * 10. mma.sync + ldmatrix + cp.async double buffering + swizzling
 */
__global__ void __launch_bounds__(NUM_THREADS)
    hgemm_mma_sync(int M, int N, int K, float alpha, const half *A,
                   const half *B, float beta, float *C) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  __trap();
#else
  __shared__ __align__(128) half As[STAGES][BM * BK];
  __shared__ __align__(128) half Bs[STAGES][BK * BN];

  const int blockRow = blockIdx.y * BM;
  const int blockCol = blockIdx.x * BN;
  const int warpIdx = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int warpRow = warpIdx / WARPS_N;
  const int warpCol = warpIdx % WARPS_N;

  float acc[MI][NJ][4] = {};

  const int numTiles = lab::ceil_div(K, BK);
  load_tile(As[0], Bs[0], 0, M, N, K, A, B, blockRow, blockCol);

  for (int t = 0; t < numTiles; ++t) {
    const int cur = t % STAGES;
    if (t + 1 < numTiles) {
      load_tile(As[(t + 1) % STAGES], Bs[(t + 1) % STAGES], (t + 1) * BK, M, N,
                K, A, B, blockRow, blockCol);
      cp_async_wait<1>();  // Tile t has landed; tile t + 1 may still be in flight.
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();

#pragma unroll
    for (int ks = 0; ks < BK / MMA_K; ++ks) {
      uint32_t a[MI][4];
      uint32_t b[NJ][2];

      // A: lanes 0-15 address rows 0-15 at the first 8 K values of this
      // step, lanes 16-31 the same rows at the next 8. The four 8 x 8
      // matrices arrive as a0a1, a2a3, a4a5, a6a7 of the A fragment.
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) {
        const int r = warpRow * WARP_TILE_M + mi * MMA_M + (lane % 16);
        const int chunk = ks * 2 + lane / 16;
        ldmatrix_x4(smem_addr(&As[cur][a_offset(r, chunk)]), a[mi][0], a[mi][1],
                    a[mi][2], a[mi][3]);
      }
      // B: lanes 0-15 address K rows 0-15 of the first n8 tile, lanes 16-31
      // the same rows of the next n8 tile. .trans turns each row-major 8 x 8
      // block into the column operand: registers 0-1 are b0b1, b2b3 of the
      // first n8 tile and registers 2-3 those of the second.
#pragma unroll
      for (int pair = 0; pair < NJ / 2; ++pair) {
        const int r = ks * MMA_K + (lane % 16);
        const int n = warpCol * WARP_TILE_N + pair * 2 * MMA_N + (lane / 16) * MMA_N;
        ldmatrix_x4_trans(smem_addr(&Bs[cur][b_offset(r, n / CHUNK)]),
                          b[2 * pair][0], b[2 * pair][1], b[2 * pair + 1][0],
                          b[2 * pair + 1][1]);
      }

#pragma unroll
      for (int mi = 0; mi < MI; ++mi) {
#pragma unroll
        for (int nj = 0; nj < NJ; ++nj) {
          mma_16816(acc[mi][nj], a[mi], b[nj]);
        }
      }
    }
    // Everyone must be done with buffer `cur` before it is refilled.
    __syncthreads();
  }

  // ---- Epilogue: C = alpha * acc + beta * C, using the C fragment layout ----
  const int g = lane / 4;
  const int tq = lane % 4;
#pragma unroll
  for (int mi = 0; mi < MI; ++mi) {
#pragma unroll
    for (int nj = 0; nj < NJ; ++nj) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int row = blockRow + warpRow * WARP_TILE_M + mi * MMA_M + g + (e / 2) * 8;
        const int col = blockCol + warpCol * WARP_TILE_N + nj * MMA_N + tq * 2 + (e % 2);
        if (row < M && col < N) {
          float &c = C[static_cast<size_t>(row) * N + col];
          c = alpha * acc[mi][nj][e] + beta * c;
        }
      }
    }
  }
#endif
}

void launch_hgemm_mma_sync(int M, int N, int K, float alpha, const half *A,
                           const half *B, float beta, float *C) {
  dim3 grid(lab::ceil_div(N, BN), lab::ceil_div(M, BM));
  hgemm_mma_sync<<<grid, NUM_THREADS>>>(M, N, K, alpha, A, B, beta, C);
}

int main(int argc, char **argv) {
  GemmRequirements req;
  req.n_multiple = 8;
  req.k_multiple = 8;
  req.min_compute_capability = 80;
  return run_gemm<half>("mma.sync + ldmatrix + cp.async", argc, argv,
                        {1024, 1024, 1024}, {144, 136, 40},
                        launch_hgemm_mma_sync, req);
}
