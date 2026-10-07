/*
 * Shared Memory Tiled Matrix Multiplication
 *
 * Intention:
 * This file introduces block tiling with shared memory to reduce redundant
 * global-memory traffic.
 *
 * High-Level Algorithm:
 * - Assign each thread block to one output tile of C.
 * - Cooperatively load one tile of A and one tile of B into shared memory.
 * - Reuse those tiles for many multiply-accumulate operations before loading
 *   the next K tile.
 * - Zero-fill tile elements that fall outside the matrices, so any M, N, and
 *   K work, including a partial last tile along K.
 *
 * The host-side driver (inputs, CPU reference, validation, timing) lives in
 * ../gemm_harness.cuh and is shared by every stage.
 */
#include "../gemm_harness.cuh"

#define TILE_SIZE 32  // Tile size for shared memory

/**
 * 2. Shared Memory Tiling Approach
 * This kernel uses shared memory to drastically reduce global memory accesses.
 * Each thread block is assigned to one tile of the output matrix C, and each
 * thread in that block computes one element within that C tile.
 * Threads within the block cooperatively load the corresponding tiles of A and
 * B into fast on-chip shared memory and reuse them for their dot products.
 * Each element loaded from global memory is now used TILE_SIZE times.
 */
__global__ void sgemm_shared(int M, int N, int K, float alpha, const float *A,
                             const float *B, float beta, float *C) {
  // Shared-memory tiles of A and B. Neither needs padding, because no warp
  // ever reads two different addresses in the same bank at the same time.
  // A warp is 32 threads with consecutive threadIdx.x and the same
  // threadIdx.y, and bank conflicts only happen between threads of one warp
  // within one instruction:
  // - Stores tileA[ty][tx] and tileB[ty][tx]: 32 consecutive words, so 32
  //   different banks.
  // - Read tileA[ty][k]: every thread reads the same word, which is a
  //   broadcast (one transaction, no conflict).
  // - Read tileB[k][tx]: 32 consecutive words of one row, so 32 different
  //   banks.
  // It is true that over the k loop each thread walks down a column of tileB,
  // but those reads happen in different instructions, so they cannot
  // conflict. Padding matters when a single warp reads down a column at once;
  // see matrix_transpose/ for that case.
  __shared__ float tileA[TILE_SIZE][TILE_SIZE];
  __shared__ float tileB[TILE_SIZE][TILE_SIZE];

  // Global row and column index in the output matrix C
  int globalRow = blockIdx.y * TILE_SIZE + threadIdx.y;
  int globalCol = blockIdx.x * TILE_SIZE + threadIdx.x;

  float partialSum = 0.0f;

  // Loop over the tiles of the shared K dimension.
  // This block owns one TILE_SIZE x TILE_SIZE output tile of C, but computing
  // that tile requires accumulating products across the full K dimension.
  // In each iteration, the block loads one tile of A and one tile of B into
  // shared memory, computes this tile's partial contribution to C, and then
  // moves to the next K tile. Rounding the tile count up (instead of K /
  // TILE_SIZE) keeps the partial last tile when K is not a multiple of
  // TILE_SIZE; its missing elements are loaded as zeros below.
  const int numTiles = lab::ceil_div(K, TILE_SIZE);
  for (int tileIdx = 0; tileIdx < numTiles; tileIdx++) {
    // Each thread loads one element of the current A tile.
    int aRow = globalRow;
    int aCol = tileIdx * TILE_SIZE + threadIdx.x;

    // Each thread also loads one element of the current B tile.
    // Over the full loop, each thread loads multiple A/B elements, one pair
    // per tileIdx iteration.
    int bRow = tileIdx * TILE_SIZE + threadIdx.y;
    int bCol = globalCol;

    // Load elements into shared memory (with bounds checking)
    if (aRow < M && aCol < K) {
      tileA[threadIdx.y][threadIdx.x] = A[aRow * K + aCol];
    } else {
      tileA[threadIdx.y][threadIdx.x] = 0.0f;
    }

    if (bRow < K && bCol < N) {
      tileB[threadIdx.y][threadIdx.x] = B[bRow * N + bCol];
    } else {
      tileB[threadIdx.y][threadIdx.x] = 0.0f;
    }

    // Wait for all threads in the block to finish loading their elements into
    // shared memory
    __syncthreads();

    // Each thread builds one tile-local dot product: over the full k loop, it
    // walks across one row of tileA and down one column of tileB.
    // At any single fixed k, though, the block collectively touches the k-th
    // column of tileA and the k-th row of tileB.
    for (int k = 0; k < TILE_SIZE; k++) {
      partialSum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
    }

    // Wait for all threads to finish computing before loading the next tile
    __syncthreads();
  }

  // Write the accumulated result to global memory, applying alpha and beta
  if (globalRow < M && globalCol < N) {
    // C = α*(A@B)+β*C
    C[globalRow * N + globalCol] =
        alpha * partialSum + beta * C[globalRow * N + globalCol];
  }
}

void launch_sgemm_shared(int M, int N, int K, float alpha, const float *A,
                         const float *B, float beta, float *C) {
  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid(lab::ceil_div(N, TILE_SIZE), lab::ceil_div(M, TILE_SIZE));
  sgemm_shared<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

int main(int argc, char **argv) {
  return run_gemm<float>("Shared memory", argc, argv, {1024, 1024, 1024},
                         {257, 129, 95}, launch_sgemm_shared);
}
