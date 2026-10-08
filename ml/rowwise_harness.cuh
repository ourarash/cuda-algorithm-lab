/*
 * rowwise_harness.cuh: shared host-side driver for row-wise ML kernels
 * (softmax, LayerNorm, RMSNorm).
 *
 * Each kernel transforms every row of a rows x cols float matrix
 * independently, the shape of activations in a transformer (rows = tokens,
 * cols = hidden size or sequence length). The harness owns random inputs
 * (and gamma/beta weights for the norms), a double-precision CPU reference,
 * and warmed-up timing reported as GB/s for the minimum traffic: read the
 * matrix once and write it once. These kernels are bandwidth-bound, so % of
 * peak bandwidth is the figure of merit.
 *
 * cols must be a multiple of 4, because the kernels use float4 accesses.
 */
#pragma once

#include <cmath>
#include <cstdio>
#include <vector>

#include "lab.cuh"

using RowwiseLauncher = void (*)(const float *d_in, float *d_out, int rows,
                                 int cols, const float *d_gamma,
                                 const float *d_beta);
using RowwiseReference = void (*)(const float *in, double *out, int rows,
                                  int cols, const float *gamma,
                                  const float *beta);

// rtol/atol: softmax outputs are tiny probabilities, so they need a near-zero
// absolute tolerance; normalized outputs are O(1) and can land near zero,
// where float rounding in the statistics needs a small absolute allowance.
inline int run_rowwise(const char *name, int argc, char **argv,
                       RowwiseLauncher launch, RowwiseReference reference,
                       double rtol, double atol) {
  lab::Args args(argc, argv);
  const int rows = static_cast<int>(args.get_int("rows", args.quick() ? 67 : 8192));
  const int cols = static_cast<int>(args.get_int("cols", args.quick() ? 1000 : 4096));
  const size_t count = static_cast<size_t>(rows) * cols;

  lab::print_device();
  std::printf("%s: %d rows x %d columns\n", name, rows, cols);
  if (cols % 4 != 0) {
    std::printf("cols must be a multiple of 4\n");
    return lab::finish(false);
  }

  const std::vector<float> input = lab::random_uniform<float>(count, -4.f, 4.f, 51);
  const std::vector<float> gamma = lab::random_uniform<float>(cols, 0.5f, 1.5f, 52);
  const std::vector<float> beta = lab::random_uniform<float>(cols, -0.5f, 0.5f, 53);
  std::vector<double> expected(count);
  reference(input.data(), expected.data(), rows, cols, gamma.data(), beta.data());

  float *d_in, *d_out, *d_gamma, *d_beta;
  CUDA_CHECK(cudaMalloc(&d_in, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_gamma, cols * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_beta, cols * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, input.data(), count * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gamma, gamma.data(), cols * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_beta, beta.data(), cols * sizeof(float),
                        cudaMemcpyHostToDevice));

  launch(d_in, d_out, rows, cols, d_gamma, d_beta);
  CUDA_CHECK_LAUNCH();
  std::vector<float> got(count);
  CUDA_CHECK(cudaMemcpy(got.data(), d_out, count * sizeof(float),
                        cudaMemcpyDeviceToHost));
  const bool pass = lab::check_close("output", got, expected, rtol, atol);

  const float ms = lab::time_ms(
      [&] { launch(d_in, d_out, rows, cols, d_gamma, d_beta); });
  lab::report(name, ms, 0, 2.0 * count * sizeof(float));

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_gamma));
  CUDA_CHECK(cudaFree(d_beta));
  return lab::finish(pass);
}

// ---- CPU references ----

inline void softmax_reference(const float *in, double *out, int rows, int cols,
                              const float *, const float *) {
  for (int r = 0; r < rows; ++r) {
    const float *x = in + static_cast<size_t>(r) * cols;
    double *y = out + static_cast<size_t>(r) * cols;
    double m = -INFINITY;
    for (int c = 0; c < cols; ++c) m = std::fmax(m, x[c]);
    double sum = 0.0;
    for (int c = 0; c < cols; ++c) sum += std::exp(x[c] - m);
    for (int c = 0; c < cols; ++c) y[c] = std::exp(x[c] - m) / sum;
  }
}

constexpr float NORM_EPS = 1e-5f;

inline void layernorm_reference(const float *in, double *out, int rows,
                                int cols, const float *gamma,
                                const float *beta) {
  for (int r = 0; r < rows; ++r) {
    const float *x = in + static_cast<size_t>(r) * cols;
    double *y = out + static_cast<size_t>(r) * cols;
    double mean = 0.0;
    for (int c = 0; c < cols; ++c) mean += x[c];
    mean /= cols;
    double var = 0.0;
    for (int c = 0; c < cols; ++c) var += (x[c] - mean) * (x[c] - mean);
    var /= cols;
    const double inv_std = 1.0 / std::sqrt(var + NORM_EPS);
    for (int c = 0; c < cols; ++c) y[c] = (x[c] - mean) * inv_std * gamma[c] + beta[c];
  }
}

inline void rmsnorm_reference(const float *in, double *out, int rows, int cols,
                              const float *gamma, const float *) {
  for (int r = 0; r < rows; ++r) {
    const float *x = in + static_cast<size_t>(r) * cols;
    double *y = out + static_cast<size_t>(r) * cols;
    double ms = 0.0;
    for (int c = 0; c < cols; ++c) ms += static_cast<double>(x[c]) * x[c];
    ms /= cols;
    const double inv_rms = 1.0 / std::sqrt(ms + NORM_EPS);
    for (int c = 0; c < cols; ++c) y[c] = x[c] * inv_rms * gamma[c];
  }
}
