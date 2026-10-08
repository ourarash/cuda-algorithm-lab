/*
 * Transpose 3: Shared-Memory Tile with Padding
 *
 * Intention:
 * Remove the 32-way bank conflict of step 2 with one extra column.
 *
 * High-Level Algorithm:
 * - Identical to step 2, except the tile is declared [32][33].
 * - With a row stride of 33 words, element tile[r][c] lives in bank
 *   (33 * r + c) % 32 = (r + c) % 32. A column read (c fixed, r = 0..31) now
 *   visits all 32 banks once, so it completes in a single transaction. Row
 *   accesses are still conflict-free.
 *
 * The cost is 32 floats of unused shared memory per tile. Step 4 gets the same
 * effect without wasting any. 03_transpose_bank_conflicts_visualization.html
 * in this folder shows the bank mapping with and without padding.
 */
#include "../transpose_harness.cuh"

__global__ void transpose_padded(const float *in, float *out, int rows, int cols) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    if (x < cols && y + j < rows) {
      tile[threadIdx.y + j][threadIdx.x] = in[static_cast<size_t>(y + j) * cols + x];
    }
  }
  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    if (x < rows && y + j < cols) {
      out[static_cast<size_t>(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
    }
  }
}

void launch(const float *in, float *out, int rows, int cols) {
  transpose_padded<<<transpose_grid(rows, cols), transpose_block()>>>(in, out, rows, cols);
}

int main(int argc, char **argv) {
  return run_transpose("3. Padded shared memory", argc, argv, launch);
}
