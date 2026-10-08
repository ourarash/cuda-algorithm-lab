/*
 * 2D Convolution 1: Filter in Constant Memory
 *
 * Intention:
 * Every thread reads the same 49 filter values in the same order. Constant
 * memory is built for exactly that: a small (64 KB) read-only space with a
 * dedicated cache that broadcasts one value to all threads of a warp in a
 * single access when they read the same address.
 *
 * Changes from step 0:
 * - The filter lives in a __constant__ array, filled from the host with
 *   cudaMemcpyToSymbol (once; it does not change between launches).
 * - The filter loops have compile-time bounds and are unrolled.
 */
#include "../convolution_harness.cuh"

__constant__ float c_filter[FILTER_WIDTH * FILTER_WIDTH];

__global__ void convolution_constant(const float *in, float *out, int width,
                                     int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  float sum = 0.0f;
#pragma unroll
  for (int fy = 0; fy < FILTER_WIDTH; ++fy) {
#pragma unroll
    for (int fx = 0; fx < FILTER_WIDTH; ++fx) {
      const int yy = y + fy - FILTER_RADIUS;
      const int xx = x + fx - FILTER_RADIUS;
      if (yy >= 0 && yy < height && xx >= 0 && xx < width) {
        sum += c_filter[fy * FILTER_WIDTH + fx] * in[yy * width + xx];
      }
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
  dim3 block(32, 8);
  dim3 grid(lab::ceil_div(width, 32), lab::ceil_div(height, 8));
  convolution_constant<<<grid, block>>>(d_in, d_out, width, height);
}

int main(int argc, char **argv) {
  return run_convolution("1. Constant-memory filter", argc, argv, launch);
}
