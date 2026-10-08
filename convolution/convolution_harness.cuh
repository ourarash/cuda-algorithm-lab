/*
 * convolution_harness.cuh: shared host-side driver for 2D convolution.
 *
 * Every step computes out = in (*) filter for a width x height float image and
 * a 7 x 7 filter, with zero padding outside the image:
 *   out[y][x] = sum over dy, dx in [-3, 3] of filter[dy + 3][dx + 3] * in[y + dy][x + dx]
 * The harness owns a random image and filter, a double-precision CPU
 * reference, and warmed-up timing reported as GFLOP/s (2 * 49 flops per
 * pixel) and GB/s (image read once, written once).
 */
#pragma once

#include <cstdio>
#include <vector>

#include "lab.cuh"

constexpr int FILTER_RADIUS = 3;
constexpr int FILTER_WIDTH = 2 * FILTER_RADIUS + 1;

// d_filter is the filter in global memory; h_filter the same values on the
// host, for steps that copy it into constant memory.
using ConvLauncher = void (*)(const float *d_in, float *d_out, int width,
                              int height, const float *d_filter,
                              const float *h_filter);

inline int run_convolution(const char *name, int argc, char **argv,
                           ConvLauncher launch) {
  lab::Args args(argc, argv);
  const int width = static_cast<int>(args.get_int("width", args.quick() ? 1000 : 4096));
  const int height = static_cast<int>(args.get_int("height", args.quick() ? 777 : 4096));
  const size_t pixels = static_cast<size_t>(width) * height;

  lab::print_device();
  std::printf("%s: %d x %d image, %d x %d filter\n", name, width, height,
              FILTER_WIDTH, FILTER_WIDTH);

  const std::vector<float> image = lab::random_uniform<float>(pixels, 0.f, 1.f, 31);
  const std::vector<float> filter =
      lab::random_uniform<float>(FILTER_WIDTH * FILTER_WIDTH, -1.f, 1.f, 32);

  std::vector<double> expected(pixels);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      double sum = 0.0;
      for (int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; ++dy) {
        for (int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; ++dx) {
          const int yy = y + dy;
          const int xx = x + dx;
          if (yy >= 0 && yy < height && xx >= 0 && xx < width) {
            sum += static_cast<double>(filter[(dy + FILTER_RADIUS) * FILTER_WIDTH +
                                              dx + FILTER_RADIUS]) *
                   image[static_cast<size_t>(yy) * width + xx];
          }
        }
      }
      expected[static_cast<size_t>(y) * width + x] = sum;
    }
  }

  float *d_in, *d_out, *d_filter;
  CUDA_CHECK(cudaMalloc(&d_in, pixels * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, pixels * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_filter, filter.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, image.data(), pixels * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_filter, filter.data(), filter.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  launch(d_in, d_out, width, height, d_filter, filter.data());
  CUDA_CHECK_LAUNCH();
  std::vector<float> got(pixels);
  CUDA_CHECK(cudaMemcpy(got.data(), d_out, pixels * sizeof(float),
                        cudaMemcpyDeviceToHost));
  const bool pass = lab::check_close("output image", got, expected, 1e-4, 1e-5);

  const float ms = lab::time_ms(
      [&] { launch(d_in, d_out, width, height, d_filter, filter.data()); });
  lab::report(name, ms, 2.0 * FILTER_WIDTH * FILTER_WIDTH * pixels,
              2.0 * pixels * sizeof(float));

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_filter));
  return lab::finish(pass);
}
