/*
 * stencil_harness.cuh: shared host-side driver for the 3D 7-point stencil.
 *
 * Every step performs one step of an explicit heat-equation (Jacobi) update
 * on an N x N x N grid of floats:
 *   out[z][y][x] = C0 * in[z][y][x]
 *                + C1 * (in[z][y][x-1] + in[z][y][x+1] + in[z][y-1][x]
 *                        + in[z][y+1][x] + in[z-1][y][x] + in[z+1][y][x])
 * for interior points; boundary points keep their input value. The harness
 * initializes the output with the input, so kernels only write the interior.
 * Validation is against a double-precision CPU reference; speed is reported
 * as GFLOP/s (8 flops per point) and GB/s (grid read once, written once).
 */
#pragma once

#include <cstdio>
#include <vector>

#include "lab.cuh"

constexpr float C0 = 0.5f;
constexpr float C1 = 1.0f / 12.0f;

using StencilLauncher = void (*)(const float *d_in, float *d_out, int n);

inline size_t grid_index(int x, int y, int z, int n) {
  return (static_cast<size_t>(z) * n + y) * n + x;
}

inline int run_stencil(const char *name, int argc, char **argv,
                       StencilLauncher launch) {
  lab::Args args(argc, argv);
  const int n = static_cast<int>(args.get_int("n", args.quick() ? 67 : 384));
  const size_t points = static_cast<size_t>(n) * n * n;

  lab::print_device();
  std::printf("%s: 7-point stencil on a %d^3 grid\n", name, n);

  const std::vector<float> grid = lab::random_uniform<float>(points, 0.f, 1.f, 41);
  std::vector<double> expected(grid.begin(), grid.end());
  for (int z = 1; z < n - 1; ++z) {
    for (int y = 1; y < n - 1; ++y) {
      for (int x = 1; x < n - 1; ++x) {
        auto at = [&](int xx, int yy, int zz) {
          return static_cast<double>(grid[grid_index(xx, yy, zz, n)]);
        };
        expected[grid_index(x, y, z, n)] =
            C0 * at(x, y, z) +
            C1 * (at(x - 1, y, z) + at(x + 1, y, z) + at(x, y - 1, z) +
                  at(x, y + 1, z) + at(x, y, z - 1) + at(x, y, z + 1));
      }
    }
  }

  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, points * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, points * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, grid.data(), points * sizeof(float),
                        cudaMemcpyHostToDevice));
  // Boundary points keep their input value; kernels only write the interior.
  CUDA_CHECK(cudaMemcpy(d_out, d_in, points * sizeof(float),
                        cudaMemcpyDeviceToDevice));

  launch(d_in, d_out, n);
  CUDA_CHECK_LAUNCH();
  std::vector<float> got(points);
  CUDA_CHECK(cudaMemcpy(got.data(), d_out, points * sizeof(float),
                        cudaMemcpyDeviceToHost));
  const bool pass = lab::check_close("output grid", got, expected, 1e-5, 1e-6);

  const float ms = lab::time_ms([&] { launch(d_in, d_out, n); });
  lab::report(name, ms, 8.0 * points, 2.0 * points * sizeof(float));

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return lab::finish(pass);
}
