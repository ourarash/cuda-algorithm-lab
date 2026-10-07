/*
 * Tensor Core Matrix Multiplication
 *
 * Intention:
 * This file demonstrates how to move from CUDA cores to NVIDIA Tensor Cores
 * through the WMMA (warp matrix multiply-accumulate) API.
 *
 * High-Level Algorithm:
 * - Each warp computes one 16x16 output tile of C. A block holds 4 warps
 *   arranged 2x2, so it covers a 32x32 region of C.
 * - Walk along K in steps of 16: load a 16x16 fragment of A and of B straight
 *   from global memory and let the Tensor Cores perform C_tile += A * B.
 * - Apply alpha and beta in registers, then store the FP32 tile.
 *
 * Precision: A and B are FP16, accumulation and C are FP32. The harness builds
 * its CPU reference from the same half-rounded inputs, so the check measures
 * the GPU's arithmetic, not the input rounding.
 *
 * Requirements:
 * - Tensor Core WMMA for FP16 needs compute capability 7.0 or newer. CUDA 13
 *   only targets 7.5 (Turing) and newer anyway.
 * - M, N, and K must be multiples of 16, because load_matrix_sync always reads
 *   a full 16x16 fragment.
 *
 * This is the simplest correct WMMA kernel, not a fast one: it has no shared
 * memory staging, so each fragment is re-read from global memory by every
 * warp that needs it. The roadmap's next steps add shared-memory tiling,
 * mma.sync with ldmatrix, and Hopper's TMA + WGMMA.
 */
#include <mma.h>

#include "../gemm_harness.cuh"

using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int WARPS_M = 2;  // Warps per block along M
constexpr int WARPS_N = 2;  // Warps per block along N
constexpr int WARP_SIZE = 32;

/**
 * 6. Hardware Acceleration (WMMA API / Tensor Cores)
 * The WMMA API programs an entire warp (32 threads) to cooperatively execute
 * a 16x16x16 mixed-precision matrix multiply-accumulate on Tensor Cores.
 */
__global__ void hgemm_wmma(int M, int N, int K, float alpha, const half *A,
                           const half *B, float beta, float *C) {
  // Which 16x16 tile of C this warp owns. All 32 threads of a warp compute
  // the same warpId, so the early return below is uniform across the warp,
  // which the *_sync WMMA calls require.
  const int warpId = threadIdx.x / WARP_SIZE;
  const int tileRow = blockIdx.y * WARPS_M + warpId / WARPS_N;
  const int tileCol = blockIdx.x * WARPS_N + warpId % WARPS_N;
  const int cRow = tileRow * WMMA_M;
  const int cCol = tileCol * WMMA_N;
  if (cRow >= M || cCol >= N) {
    return;
  }

  // Fragments are register tiles distributed across the 32 threads of the
  // warp. matrix_a and matrix_b hold FP16 inputs; the accumulator is FP32.
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
      b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
  wmma::fill_fragment(acc_frag, 0.0f);

  for (int k = 0; k < K; k += WMMA_K) {
    // Load 16x16 tiles from global memory directly into the fragments. The
    // last argument is the leading dimension (row length) of the matrix.
    wmma::load_matrix_sync(a_frag, A + cRow * K + k, K);
    wmma::load_matrix_sync(b_frag, B + k * N + cCol, N);

    // Tensor Core multiply-accumulate: acc += A_tile * B_tile.
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  // C = alpha * acc + beta * C. The mapping of fragment elements to matrix
  // positions is unspecified, but it is the same for two fragments of the
  // same type, so element-wise math between them is valid.
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  wmma::load_matrix_sync(c_frag, C + cRow * N + cCol, N, wmma::mem_row_major);
  for (int i = 0; i < c_frag.num_elements; i++) {
    c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
  }
  wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
}

void launch_hgemm_wmma(int M, int N, int K, float alpha, const half *A,
                       const half *B, float beta, float *C) {
  dim3 block(WARPS_M * WARPS_N * WARP_SIZE);  // 4 warps = 128 threads
  dim3 grid(lab::ceil_div(N, WARPS_N * WMMA_N), lab::ceil_div(M, WARPS_M * WMMA_M));
  hgemm_wmma<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

int main(int argc, char **argv) {
  GemmRequirements req;
  req.m_multiple = WMMA_M;
  req.n_multiple = WMMA_N;
  req.k_multiple = WMMA_K;
  // The quick shape leaves some warps without a tile (M = 144 is 4.5 blocks
  // of 32 rows), which exercises the early return.
  return run_gemm<half>("Tensor cores (WMMA)", argc, argv, {1024, 1024, 1024},
                        {144, 80, 48}, launch_hgemm_wmma, req);
}
