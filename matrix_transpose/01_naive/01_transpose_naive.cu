/*
 * Transpose 1: Naive
 *
 * Intention:
 * The direct translation of out[c][r] = in[r][c] into a kernel, to show why a
 * transpose cannot be fast without help.
 *
 * High-Level Algorithm:
 * - Same tiling and thread layout as the copy in step 0.
 * - Each thread reads in[y][x] and writes out[x][y].
 *
 * What is slow:
 * Reads are coalesced: consecutive threads read consecutive x. Writes are not:
 * consecutive threads write out[x][y], out[x + 1][y], ..., addresses `rows`
 * floats apart, so every thread of a warp writes to a different 32-byte
 * sector. The memory system moves whole sectors, so most of the write
 * bandwidth is wasted.
 */
#include "../transpose_harness.cuh"

__global__ void transpose_naive(const float *in, float *out, int rows, int cols) {
  const int x = blockIdx.x * TILE_DIM + threadIdx.x;
  const int y = blockIdx.y * TILE_DIM + threadIdx.y;
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    if (x < cols && y + j < rows) {
      out[static_cast<size_t>(x) * rows + (y + j)] =
          in[static_cast<size_t>(y + j) * cols + x];
    }
  }
}

void launch(const float *in, float *out, int rows, int cols) {
  transpose_naive<<<transpose_grid(rows, cols), transpose_block()>>>(in, out, rows, cols);
}

int main(int argc, char **argv) {
  return run_transpose("1. Naive", argc, argv, launch);
}
