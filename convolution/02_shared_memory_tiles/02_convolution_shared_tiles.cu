/*
 * 2D Convolution 2: Shared-Memory Input Tiles with a Halo
 *
 * Intention:
 * A block of threads computes a 32 x 8 tile of output pixels, and together
 * they need a (32 + 6) x (8 + 6) window of the input: the tile plus a
 * 3-pixel "halo" on every side. Load that window into shared memory once,
 * then let every thread read its 7 x 7 neighbourhood from shared memory
 * instead of global memory.
 *
 * High-Level Algorithm:
 * - Cooperative load: the 256 threads walk the 38 x 14 window together
 *   (two or three elements each), writing zeros for positions outside the
 *   image, which implements the zero padding once instead of in the inner
 *   loop.
 * - __syncthreads().
 * - Each thread computes its output pixel from shared memory and the
 *   constant-memory filter, with no bounds checks in the inner loop.
 *
 * A block now loads 38 x 14 = 532 input values for its 256 outputs, so each
 * input value is read from global memory about 2.1 times (the halos of
 * neighbouring blocks overlap) instead of up to 49 times. The larger the tile
 * relative to the filter, the closer that gets to 1.
 */
#include "../convolution_harness.cuh"

constexpr int TILE_X = 32;
constexpr int TILE_Y = 8;
constexpr int IN_X = TILE_X + 2 * FILTER_RADIUS;  // 38
constexpr int IN_Y = TILE_Y + 2 * FILTER_RADIUS;  // 14

__constant__ float c_filter[FILTER_WIDTH * FILTER_WIDTH];

__global__ void convolution_shared_tiles(const float *in, float *out,
                                         int width, int height) {
  __shared__ float tile[IN_Y][IN_X];

  // Top-left corner of the input window this block needs.
  const int in_x0 = blockIdx.x * TILE_X - FILTER_RADIUS;
  const int in_y0 = blockIdx.y * TILE_Y - FILTER_RADIUS;
  const int tid = threadIdx.y * TILE_X + threadIdx.x;
  for (int idx = tid; idx < IN_X * IN_Y; idx += TILE_X * TILE_Y) {
    const int ty = idx / IN_X;
    const int tx = idx % IN_X;
    const int gx = in_x0 + tx;
    const int gy = in_y0 + ty;
    tile[ty][tx] = (gx >= 0 && gx < width && gy >= 0 && gy < height)
                       ? in[gy * width + gx]
                       : 0.0f;
  }
  __syncthreads();

  const int x = blockIdx.x * TILE_X + threadIdx.x;
  const int y = blockIdx.y * TILE_Y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  float sum = 0.0f;
#pragma unroll
  for (int fy = 0; fy < FILTER_WIDTH; ++fy) {
#pragma unroll
    for (int fx = 0; fx < FILTER_WIDTH; ++fx) {
      sum += c_filter[fy * FILTER_WIDTH + fx] *
             tile[threadIdx.y + fy][threadIdx.x + fx];
    }
  }
  out[y * width + x] = sum;
}

void launch(const float *d_in, float *d_out, int width, int height,
            const float *, const float *h_filter) {
  static bool uploaded = false;
  if (!uploaded) {
    CUDA_CHECK(cudaMemcpyToSymbol(c_filter, h_filter,
                                  FILTER_WIDTH * FILTER_WIDTH * sizeof(float)));
    uploaded = true;
  }
  dim3 block(TILE_X, TILE_Y);
  dim3 grid(lab::ceil_div(width, TILE_X), lab::ceil_div(height, TILE_Y));
  convolution_shared_tiles<<<grid, block>>>(d_in, d_out, width, height);
}

int main(int argc, char **argv) {
  return run_convolution("2. Shared-memory tiles + halo", argc, argv, launch);
}
