/*
 * Uncoalesced Matrix Multiplication
 *
 * Intention:
 * This file intentionally demonstrates a poor thread-to-data mapping so the
 * memory-access anti-pattern is easy to see.
 *
 * High-Level Algorithm:
 * - Launch a 2D thread block.
 * - Let each thread compute one output element C[i, j].
 * - Map threadIdx.x to rows instead of columns, which makes neighboring
 *   threads walk through memory with a large stride.
 * - Use this version as the baseline that later kernels improve upon.
 *
 * The host-side driver (inputs, CPU reference, validation, timing) lives in
 * ../gemm_harness.cuh and is shared by every stage.
 */
#include "../gemm_harness.cuh"

constexpr int BLOCK_SIZE = 32;

/**
 * 0. The Anti-Pattern: Uncoalesced Memory Access
 * This kernel maps the fastest-changing thread index (threadIdx.x) to matrix
 * rows. Because memory is row-major, adjacent threads access memory locations
 * that are far apart, leading to a catastrophic loss in memory bandwidth.
 */
__global__ void sgemm_uncoalesced(int M, int N, int K, float alpha,
                                  const float *A, const float *B, float beta,
                                  float *C) {
  // Compute the position in C that this thread is responsible for.
  // Note that i changes with threadIdx.x and j with threadIdx.y,
  // so i changes faster than j across a warp. Therefore:
  // - A[i, k] is not coalesced (adjacent threads access different rows)
  // - B[k, j] is broadcast (k and j are the same for the whole warp)
  // - C[i, j] is not coalesced (adjacent threads access different rows)
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  // The `if` is necessary when M or N is not a multiple of 32.
  if (i < M && j < N) {
    float tmp = 0.0f;
    for (int k = 0; k < K; ++k) {
      tmp += A[i * K + k] * B[k * N + j];  // A[i, k] * B[k, j]
    }
    // C = α*(A@B)+β*C
    C[i * N + j] = alpha * tmp + beta * C[i * N + j];
  }
}

void launch_sgemm_uncoalesced(int M, int N, int K, float alpha, const float *A,
                              const float *B, float beta, float *C) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(lab::ceil_div(M, BLOCK_SIZE), lab::ceil_div(N, BLOCK_SIZE));
  sgemm_uncoalesced<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

int main(int argc, char **argv) {
  return run_gemm<float>("Uncoalesced", argc, argv, {1024, 1024, 1024},
                         {257, 129, 95}, launch_sgemm_uncoalesced);
}
