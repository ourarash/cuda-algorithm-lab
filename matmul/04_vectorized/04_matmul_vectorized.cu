/*
 * Vectorized Matrix Multiplication
 *
 * Intention:
 * This file keeps the 2D register tiling from the previous stage and widens
 * its memory instructions to 128 bits with float4, so the same data moves
 * with a quarter of the load/store instructions.
 *
 * High-Level Algorithm:
 * - Same block tile (128 x 128), K tile (8), and 8 x 8 outputs per thread as
 *   03_register_tiling/04_matmul_2d_register_tiling.cu.
 * - Global -> shared: each thread loads one float4 of A and one float4 of B
 *   per K tile instead of four scalar floats of each.
 * - Store the A tile transposed in shared memory (As[k][m]). Each thread's 8
 *   A values for a given k are then contiguous, so they can be read with two
 *   float4 loads, exactly like its 8 B values.
 * - Read and write C with float4 as well.
 *
 * Requirements:
 * float4 accesses must be 16-byte aligned. cudaMalloc returns aligned
 * pointers, and every float4 here starts at a column that is a multiple of 4,
 * so rows must also start on a 16-byte boundary: N and K must be multiples of
 * 4. M can be anything. The harness rejects other sizes.
 *
 * The host-side driver (inputs, CPU reference, validation, timing) lives in
 * ../gemm_harness.cuh and is shared by every stage.
 */
#include "../gemm_harness.cuh"

constexpr int BM = 128;  // Block tile size in M
constexpr int BN = 128;  // Block tile size in N
constexpr int BK = 8;    // Block tile size in K
constexpr int TM = 8;    // Outputs per thread in M
constexpr int TN = 8;    // Outputs per thread in N
constexpr int NUM_THREADS = (BM / TM) * (BN / TN);  // 256

// Each K tile of A (BM x BK) and of B (BK x BN) holds exactly one float4 per
// thread, which keeps the loading code free of loops.
static_assert(BM * BK == 4 * NUM_THREADS, "one float4 of A per thread");
static_assert(BK * BN == 4 * NUM_THREADS, "one float4 of B per thread");

__device__ __forceinline__ float4 load_float4(const float *p) {
  return *reinterpret_cast<const float4 *>(p);
}

/**
 * 5. Vectorized memory access (float4) on top of 2D register tiling.
 */
__global__ void sgemm_vectorized(int M, int N, int K, float alpha,
                                 const float *A, const float *B, float beta,
                                 float *C) {
  // As is stored transposed: As[k][m] holds A[blockRow + m][k0 + k].
  // __align__(16) guarantees the float4 reads below are aligned; a plain
  // float array is only guaranteed 4-byte alignment.
  __shared__ __align__(16) float As[BK][BM];
  __shared__ __align__(16) float Bs[BK][BN];

  const int blockRow = blockIdx.y * BM;
  const int blockCol = blockIdx.x * BN;

  // Position of this thread's 8 x 8 output patch inside the block tile.
  const int threadRow = threadIdx.x / (BN / TN);  // 0..15
  const int threadCol = threadIdx.x % (BN / TN);  // 0..15

  // Which float4 of the A tile and of the B tile this thread loads.
  const int aRow = threadIdx.x / (BK / 4);        // 0..127
  const int aCol = (threadIdx.x % (BK / 4)) * 4;  // 0 or 4
  const int bRow = threadIdx.x / (BN / 4);        // 0..7
  const int bCol = (threadIdx.x % (BN / 4)) * 4;  // 0, 4, ..., 124

  float acc[TM][TN] = {};
  float regM[TM];
  float regN[TN];

  for (int k0 = 0; k0 < K; k0 += BK) {
    // ---- Global -> shared, one 128-bit load each for A and B. ----
    // Since K % 4 == 0 and aCol % 4 == 0, either all four values of the
    // float4 are inside the matrix or none are, so one bounds check covers
    // the whole vector. The same holds for B with N % 4 == 0.
    const int gRowA = blockRow + aRow;
    const int gColA = k0 + aCol;
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
    if (gRowA < M && gColA < K) {
      a = load_float4(&A[static_cast<size_t>(gRowA) * K + gColA]);
    }
    // Transpose while storing: the four consecutive K values of one A row go
    // to four different rows of As.
    As[aCol + 0][aRow] = a.x;
    As[aCol + 1][aRow] = a.y;
    As[aCol + 2][aRow] = a.z;
    As[aCol + 3][aRow] = a.w;

    const int gRowB = k0 + bRow;
    const int gColB = blockCol + bCol;
    float4 b = make_float4(0.f, 0.f, 0.f, 0.f);
    if (gRowB < K && gColB < N) {
      b = load_float4(&B[static_cast<size_t>(gRowB) * N + gColB]);
    }
    *reinterpret_cast<float4 *>(&Bs[bRow][bCol]) = b;

    __syncthreads();

    // ---- Shared -> registers -> 64 FMAs per k, as in the previous stage. ----
#pragma unroll
    for (int k = 0; k < BK; ++k) {
      // Thanks to the transposed layout, this thread's 8 A values for this k
      // are contiguous: two float4 loads instead of eight scalar loads.
#pragma unroll
      for (int i = 0; i < TM; i += 4) {
        const float4 t = load_float4(&As[k][threadRow * TM + i]);
        regM[i + 0] = t.x;
        regM[i + 1] = t.y;
        regM[i + 2] = t.z;
        regM[i + 3] = t.w;
      }
#pragma unroll
      for (int j = 0; j < TN; j += 4) {
        const float4 t = load_float4(&Bs[k][threadCol * TN + j]);
        regN[j + 0] = t.x;
        regN[j + 1] = t.y;
        regN[j + 2] = t.z;
        regN[j + 3] = t.w;
      }
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          acc[i][j] += regM[i] * regN[j];
        }
      }
    }

    __syncthreads();
  }

  // ---- Epilogue: C = alpha * acc + beta * C with float4 reads and writes. ----
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = blockRow + threadRow * TM + i;
#pragma unroll
    for (int j = 0; j < TN; j += 4) {
      const int col = blockCol + threadCol * TN + j;
      if (row < M && col < N) {  // N % 4 == 0, so col < N covers col + 3.
        float4 *c_ptr =
            reinterpret_cast<float4 *>(&C[static_cast<size_t>(row) * N + col]);
        float4 c = *c_ptr;
        c.x = alpha * acc[i][j + 0] + beta * c.x;
        c.y = alpha * acc[i][j + 1] + beta * c.y;
        c.z = alpha * acc[i][j + 2] + beta * c.z;
        c.w = alpha * acc[i][j + 3] + beta * c.w;
        *c_ptr = c;
      }
    }
  }
}

void launch_sgemm_vectorized(int M, int N, int K, float alpha, const float *A,
                             const float *B, float beta, float *C) {
  dim3 grid(lab::ceil_div(N, BN), lab::ceil_div(M, BM));
  sgemm_vectorized<<<grid, NUM_THREADS>>>(M, N, K, alpha, A, B, beta, C);
}

int main(int argc, char **argv) {
  GemmRequirements req;
  req.n_multiple = 4;
  req.k_multiple = 4;
  return run_gemm<float>("Vectorized (float4)", argc, argv, {1024, 1024, 1024},
                         {257, 132, 100}, launch_sgemm_vectorized, req);
}
