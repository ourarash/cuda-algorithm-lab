/*
 * Warptiled Matrix Multiplication
 *
 * Intention:
 * This file adds a third level of tiling between the thread block and the
 * thread: the warp. Each warp owns a contiguous 64 x 64 piece of the block's
 * output tile, and each thread computes several small sub-tiles inside it.
 *
 * Tiling hierarchy (outermost to innermost):
 * - Block tile:  128 x 128 outputs per thread block (4 warps, 128 threads).
 * - Warp tile:   64 x 64 outputs per warp; the block's 4 warps form a 2 x 2
 *                grid.
 * - Warp subtile: the warp tile is split into WNITER = 4 subtiles of 64 x 16
 *                along N. In one subtile, the 32 threads form an 8 x 4 grid.
 * - Thread tile: TM x TN = 8 x 4 outputs per thread per subtile, so each
 *                thread accumulates 4 x 8 x 4 = 128 outputs in registers.
 *
 * Why the warp level matters:
 * - Locality: in the previous stage the 256 threads of a block were laid out
 *   across the whole 128 x 128 tile, so one warp touched a wide, thin strip.
 *   Here a warp's threads all work inside one compact 64 x 64 region, so the
 *   shared-memory values a warp needs in one step are few and are reused by
 *   many of its threads in the same instruction (broadcasts).
 * - More work per shared-memory read: per k step each thread reads 24 floats
 *   from shared memory and performs 128 FMAs (5.3 per float), up from 16
 *   floats and 64 FMAs (4 per float) in the vectorized stage.
 * - It mirrors how Tensor Core code is organized (stages 08-11), where the
 *   warp is the unit that executes a matrix instruction.
 *
 * Shared-memory layout and loads are the same as the vectorized stage:
 * float4 global loads, A stored transposed (As[k][m]) so each thread's TM A
 * values are contiguous, float4 shared-memory reads, float4 epilogue.
 * Requirements: N and K must be multiples of 4 (16-byte aligned rows).
 *
 * Tile sizes follow kernel 10 of Simon Boehm's "How to Optimize a CUDA Matmul
 * Kernel for cuBLAS-like Performance".
 *
 * The host-side driver (inputs, CPU reference, validation, timing, cuBLAS
 * baseline) lives in ../gemm_harness.cuh and is shared by every stage.
 */
#include "../gemm_harness.cuh"

constexpr int NUM_THREADS = 128;
constexpr int BM = 128;  // Block tile
constexpr int BN = 128;
constexpr int BK = 16;
constexpr int WM = 64;  // Warp tile
constexpr int WN = 64;
constexpr int WNITER = 4;  // Warp subtiles along N
constexpr int TM = 8;      // Thread tile
constexpr int TN = 4;
constexpr int WARP_SIZE = 32;

// Derived sizes.
constexpr int WMITER = (WM * WN) / (WARP_SIZE * TM * TN * WNITER);  // 1
constexpr int WSUBM = WM / WMITER;                                  // 64
constexpr int WSUBN = WN / WNITER;                                  // 16

static_assert((BM / WM) * (BN / WN) * WARP_SIZE == NUM_THREADS,
              "one warp per warp tile");
static_assert((WSUBM / TM) * (WSUBN / TN) == WARP_SIZE,
              "a warp's threads exactly cover one warp subtile");
static_assert((BM * BK) % (4 * NUM_THREADS) == 0, "A tile loads evenly");
static_assert((BK * BN) % (4 * NUM_THREADS) == 0, "B tile loads evenly");
static_assert(TM % 4 == 0 && TN % 4 == 0, "float4 shared-memory reads");

__device__ __forceinline__ float4 load_float4(const float *p) {
  return *reinterpret_cast<const float4 *>(p);
}

/**
 * 6. Warptiling
 */
__global__ void sgemm_warptiling(int M, int N, int K, float alpha,
                                 const float *A, const float *B, float beta,
                                 float *C) {
  __shared__ __align__(16) float As[BK][BM];  // Transposed: As[k][m]
  __shared__ __align__(16) float Bs[BK][BN];

  const int blockRow = blockIdx.y * BM;
  const int blockCol = blockIdx.x * BN;

  // Warp position inside the block tile.
  const int warpIdx = threadIdx.x / WARP_SIZE;
  const int warpRow = warpIdx / (BN / WN);
  const int warpCol = warpIdx % (BN / WN);

  // Thread position inside one warp subtile (an 8 x 4 grid of threads).
  const int lane = threadIdx.x % WARP_SIZE;
  const int threadRowInWarp = lane / (WSUBN / TN);
  const int threadColInWarp = lane % (WSUBN / TN);

  // Global -> shared load mapping: consecutive threads read consecutive
  // float4s of a row, so the loads are coalesced.
  const int innerRowA = threadIdx.x / (BK / 4);
  const int innerColA = (threadIdx.x % (BK / 4)) * 4;
  constexpr int rowStrideA = NUM_THREADS / (BK / 4);
  const int innerRowB = threadIdx.x / (BN / 4);
  const int innerColB = (threadIdx.x % (BN / 4)) * 4;
  constexpr int rowStrideB = NUM_THREADS / (BN / 4);

  float acc[WMITER * TM][WNITER * TN] = {};
  float regM[WMITER * TM];
  float regN[WNITER * TN];

  for (int k0 = 0; k0 < K; k0 += BK) {
    // ---- Global -> shared ----
    for (int offset = 0; offset < BM; offset += rowStrideA) {
      const int row = blockRow + innerRowA + offset;
      const int col = k0 + innerColA;
      float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
      if (row < M && col < K) {  // K % 4 == 0: the whole float4 is in range
        a = load_float4(&A[static_cast<size_t>(row) * K + col]);
      }
      As[innerColA + 0][innerRowA + offset] = a.x;
      As[innerColA + 1][innerRowA + offset] = a.y;
      As[innerColA + 2][innerRowA + offset] = a.z;
      As[innerColA + 3][innerRowA + offset] = a.w;
    }
    for (int offset = 0; offset < BK; offset += rowStrideB) {
      const int row = k0 + innerRowB + offset;
      const int col = blockCol + innerColB;
      float4 b = make_float4(0.f, 0.f, 0.f, 0.f);
      if (row < K && col < N) {  // N % 4 == 0
        b = load_float4(&B[static_cast<size_t>(row) * N + col]);
      }
      *reinterpret_cast<float4 *>(&Bs[innerRowB + offset][innerColB]) = b;
    }
    __syncthreads();

    // ---- Shared -> registers -> outer products ----
#pragma unroll
    for (int k = 0; k < BK; ++k) {
#pragma unroll
      for (int wSubRow = 0; wSubRow < WMITER; ++wSubRow) {
#pragma unroll
        for (int i = 0; i < TM; i += 4) {
          const float4 t = load_float4(
              &As[k][warpRow * WM + wSubRow * WSUBM + threadRowInWarp * TM + i]);
          regM[wSubRow * TM + i + 0] = t.x;
          regM[wSubRow * TM + i + 1] = t.y;
          regM[wSubRow * TM + i + 2] = t.z;
          regM[wSubRow * TM + i + 3] = t.w;
        }
      }
#pragma unroll
      for (int wSubCol = 0; wSubCol < WNITER; ++wSubCol) {
#pragma unroll
        for (int j = 0; j < TN; j += 4) {
          const float4 t = load_float4(
              &Bs[k][warpCol * WN + wSubCol * WSUBN + threadColInWarp * TN + j]);
          regN[wSubCol * TN + j + 0] = t.x;
          regN[wSubCol * TN + j + 1] = t.y;
          regN[wSubCol * TN + j + 2] = t.z;
          regN[wSubCol * TN + j + 3] = t.w;
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
          if (row < M && col < N) {  // N % 4 == 0 covers col + 3
            float4 *c_ptr = reinterpret_cast<float4 *>(
                &C[static_cast<size_t>(row) * N + col]);
            // Index acc directly (not through a pointer) so the compiler can
            // keep the whole array in registers.
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

void launch_sgemm_warptiling(int M, int N, int K, float alpha, const float *A,
                             const float *B, float beta, float *C) {
  dim3 grid(lab::ceil_div(N, BN), lab::ceil_div(M, BM));
  sgemm_warptiling<<<grid, NUM_THREADS>>>(M, N, K, alpha, A, B, beta, C);
}

int main(int argc, char **argv) {
  GemmRequirements req;
  req.n_multiple = 4;
  req.k_multiple = 4;
  return run_gemm<float>("Warptiling", argc, argv, {1024, 1024, 1024},
                         {257, 132, 100}, launch_sgemm_warptiling, req);
}
