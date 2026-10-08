/*
 * Transpose 0: Copy (the Speed Limit)
 *
 * Intention:
 * A transpose reads every element once and writes it once, exactly like a
 * copy. A copy kernel with the same tiling and thread layout, but without
 * swapping coordinates, therefore shows the best bandwidth a transpose could
 * hope for on this GPU. Every later step is compared against this number.
 *
 * High-Level Algorithm:
 * - Each block covers one 32 x 32 tile; each of its 32 x 8 threads copies 4
 *   elements, rows threadIdx.y, +8, +16, +24.
 * - Consecutive threads (threadIdx.x) touch consecutive columns, so reads and
 *   writes are both fully coalesced.
 */
#include "../transpose_harness.cuh"

__global__ void copy_tiled(const float *in, float *out, int rows, int cols) {
  const int x = blockIdx.x * TILE_DIM + threadIdx.x;
  const int y = blockIdx.y * TILE_DIM + threadIdx.y;
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    if (x < cols && y + j < rows) {
      out[static_cast<size_t>(y + j) * cols + x] =
          in[static_cast<size_t>(y + j) * cols + x];
    }
  }
}

void launch(const float *in, float *out, int rows, int cols) {
  copy_tiled<<<transpose_grid(rows, cols), transpose_block()>>>(in, out, rows, cols);
}

int main(int argc, char **argv) {
  return run_transpose("0. Copy (speed limit)", argc, argv, launch, /*is_copy=*/true);
}
