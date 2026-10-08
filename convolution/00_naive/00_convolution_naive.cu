/*
 * 2D Convolution 0: Naive
 *
 * Intention:
 * One thread per output pixel, reading its 7 x 7 neighbourhood and the filter
 * straight from global memory. Simple and correct; the baseline for the next
 * two steps.
 *
 * High-Level Algorithm:
 * - 2D grid of 32 x 8 thread blocks over the output image.
 * - Each thread sums filter[dy][dx] * in[y + dy][x + dx], skipping positions
 *   outside the image (zero padding).
 *
 * What is wasteful:
 * - Every thread reads all 49 filter values from global memory, although all
 *   threads read the same 49 values.
 * - Neighbouring threads read heavily overlapping input windows; the caches
 *   catch much of that, but each value is still requested up to 49 times.
 */
#include "../convolution_harness.cuh"

__global__ void convolution_naive(const float *in, float *out, int width,
                                  int height, const float *filter) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  float sum = 0.0f;
  for (int fy = 0; fy < FILTER_WIDTH; ++fy) {
    for (int fx = 0; fx < FILTER_WIDTH; ++fx) {
      const int yy = y + fy - FILTER_RADIUS;
      const int xx = x + fx - FILTER_RADIUS;
      if (yy >= 0 && yy < height && xx >= 0 && xx < width) {
        sum += filter[fy * FILTER_WIDTH + fx] * in[yy * width + xx];
      }
    }
  }
  out[y * width + x] = sum;
}

void launch(const float *d_in, float *d_out, int width, int height,
            const float *d_filter, const float *) {
  dim3 block(32, 8);
  dim3 grid(lab::ceil_div(width, 32), lab::ceil_div(height, 8));
  convolution_naive<<<grid, block>>>(d_in, d_out, width, height, d_filter);
}

int main(int argc, char **argv) {
  return run_convolution("0. Naive", argc, argv, launch);
}
