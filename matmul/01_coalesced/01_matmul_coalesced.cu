/*
 * Coalesced Matrix Multiplication
 *
 * Intention:
 * This file fixes the main flaw in the uncoalesced version by changing the
 * thread mapping so neighboring threads access neighboring columns.
 *
 * High-Level Algorithm:
 * - Launch a 2D thread block.
 * - Let each thread compute one output element C[i, j].
 * - Map threadIdx.x to columns so loads from B and stores to C become
 *   coalesced across the warp.
 * - Keep everything else simple so the effect of thread mapping is isolated.
 *
 * The host-side driver (inputs, CPU reference, validation, timing) lives in
 * ../gemm_harness.cuh and is shared by every stage.
 */
#include "../gemm_harness.cuh"

constexpr int BLOCK_SIZE = 32;

/**
 * 1. The First Fix: Coalesced Memory Access
 * This kernel maps the fastest-changing thread index (threadIdx.x) to matrix
 * columns. Adjacent threads now access adjacent memory locations, allowing the
 * GPU to "coalesce" these reads into a single, efficient transaction.
 */
__global__ void sgemm_coalesced(int M, int N, int K, float alpha,
                                const float *A, const float *B, float beta,
                                float *C) {
  // Compute the position in C that this thread is responsible for.
  // Note that j changes with threadIdx.x and i with threadIdx.y,
  // so j changes faster than i across a warp. Therefore:
  // - A[i, k] is broadcast (i and k are the same for the whole warp)
  // - B[k, j] is coalesced (adjacent threads access adjacent columns)
  // - C[i, j] is coalesced (adjacent threads access adjacent columns)
  //
  // An equivalent formulation (used in Simon Boehm's GEMM walkthrough) is to
  // launch a 1D block of 32 * 32 threads and recover the position from the
  // flattened thread id:
  //   i = blockIdx.y * 32 + threadIdx.x / 32;
  //   j = blockIdx.x * 32 + threadIdx.x % 32;
  // What matters is the same in both: consecutive thread ids map to
  // consecutive columns.
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = blockIdx.y * blockDim.y + threadIdx.y;

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

void launch_sgemm_coalesced(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(lab::ceil_div(N, BLOCK_SIZE), lab::ceil_div(M, BLOCK_SIZE));
  sgemm_coalesced<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

int main(int argc, char **argv) {
  return run_gemm<float>("Coalesced", argc, argv, {1024, 1024, 1024},
                         {257, 129, 95}, launch_sgemm_coalesced);
}
