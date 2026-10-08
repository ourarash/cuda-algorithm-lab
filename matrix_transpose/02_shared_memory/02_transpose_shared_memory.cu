/*
 * Transpose 2: Shared-Memory Tile
 *
 * Intention:
 * Make both the reads and the writes coalesced by staging each tile in shared
 * memory, where the transposition happens instead.
 *
 * High-Level Algorithm:
 * - Read a 32 x 32 tile row by row (coalesced) into tile[ty][tx].
 * - __syncthreads().
 * - Write the transposed tile row by row to the mirrored block position:
 *   the thread that writes out[...][x] reads tile[tx][ty], a column of the
 *   shared tile. Consecutive threads now write consecutive addresses.
 *
 * What is slow now:
 * Shared memory has 32 banks of 4-byte words, and a 32-float row spans each
 * bank exactly once. A column read, tile[tx][ty] for tx = 0..31, hits the
 * same bank 32 times: a 32-way bank conflict, so each such read is
 * serialized into 32 transactions. Steps 3 and 4 fix this.
 */
#include "../transpose_harness.cuh"

__global__ void transpose_shared(const float *in, float *out, int rows, int cols) {
  __shared__ float tile[TILE_DIM][TILE_DIM];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    if (x < cols && y + j < rows) {
      tile[threadIdx.y + j][threadIdx.x] = in[static_cast<size_t>(y + j) * cols + x];
    }
  }
  __syncthreads();

  // The output block sits at the mirrored position of the input block.
  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    if (x < rows && y + j < cols) {
      out[static_cast<size_t>(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
    }
  }
}

void launch(const float *in, float *out, int rows, int cols) {
  transpose_shared<<<transpose_grid(rows, cols), transpose_block()>>>(in, out, rows, cols);
}

int main(int argc, char **argv) {
  return run_transpose("2. Shared memory", argc, argv, launch);
}
