/*
 * Transpose 4: Shared-Memory Tile with an XOR Swizzle
 *
 * Intention:
 * Avoid bank conflicts without padding by permuting where each element of the
 * tile is stored. This is the technique high-performance kernels use (the
 * GEMM in matmul/10_mma_sync and Hopper's TMA hardware both swizzle shared
 * memory), because it keeps tiles dense and aligned.
 *
 * High-Level Algorithm:
 * - Store logical element (r, c) of the tile at tile[r][c ^ r].
 * - Row write by a warp: r fixed, c = 0..31, so c ^ r takes every column
 *   value once: 32 banks, no conflict.
 * - Column read by a warp: c fixed, r = 0..31, so the physical column c ^ r
 *   again takes every value once, and each lands in a different bank.
 * - XOR with r is its own inverse, so reading logical (r, c) back is just
 *   tile[r][c ^ r] again. The tile stays 32 x 32: no wasted memory.
 */
#include "../transpose_harness.cuh"

static_assert((TILE_DIM & (TILE_DIM - 1)) == 0,
              "XOR swizzling keeps indices in range for power-of-two tiles");

__global__ void transpose_swizzled(const float *in, float *out, int rows, int cols) {
  __shared__ float tile[TILE_DIM][TILE_DIM];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    const int r = threadIdx.y + j;
    const int c = threadIdx.x;
    if (x < cols && y + j < rows) {
      tile[r][c ^ r] = in[static_cast<size_t>(y + j) * cols + x];
    }
  }
  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    // This thread needs logical element (r, c) = (threadIdx.x, threadIdx.y + j).
    const int r = threadIdx.x;
    const int c = threadIdx.y + j;
    if (x < rows && y + j < cols) {
      out[static_cast<size_t>(y + j) * rows + x] = tile[r][c ^ r];
    }
  }
}

void launch(const float *in, float *out, int rows, int cols) {
  transpose_swizzled<<<transpose_grid(rows, cols), transpose_block()>>>(in, out, rows, cols);
}

int main(int argc, char **argv) {
  return run_transpose("4. Swizzled shared memory", argc, argv, launch);
}
