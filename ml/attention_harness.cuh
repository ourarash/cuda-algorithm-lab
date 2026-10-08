/*
 * attention_harness.cuh: shared host-side driver for scaled dot-product
 * attention (steps 04-05).
 *
 * For each of bh independent heads (batch x heads) with sequence length n and
 * head dimension 64:
 *   O = softmax(Q K^T / sqrt(64)) V,   Q, K, V, O: n x 64, row-major
 * (no causal mask). Tensors are stored [bh][n][64]. The harness owns random
 * inputs, a double-precision CPU reference, and warmed-up timing reported as
 * GFLOP/s (4 * n * n * 64 flops per head: Q K^T and P V) and as GB/s of the
 * minimum traffic (read Q, K, V once, write O once).
 */
#pragma once

#include <cmath>
#include <cstdio>
#include <vector>

#include "lab.cuh"

constexpr int HEAD_DIM = 64;

// d_scratch holds at least bh * n * n floats (enough for a full score matrix).
using AttentionLauncher = void (*)(const float *q, const float *k,
                                   const float *v, float *o, int bh, int n,
                                   float *d_scratch);

inline int run_attention(const char *name, int argc, char **argv,
                         AttentionLauncher launch) {
  lab::Args args(argc, argv);
  const int bh = static_cast<int>(args.get_int("heads", args.quick() ? 2 : 16));
  const int n = static_cast<int>(args.get_int("seq", args.quick() ? 200 : 1024));
  const size_t elems = static_cast<size_t>(bh) * n * HEAD_DIM;

  lab::print_device();
  std::printf("%s: %d heads, sequence length %d, head dim %d\n", name, bh, n,
              HEAD_DIM);

  const std::vector<float> q = lab::random_uniform<float>(elems, -1.f, 1.f, 61);
  const std::vector<float> k = lab::random_uniform<float>(elems, -1.f, 1.f, 62);
  const std::vector<float> v = lab::random_uniform<float>(elems, -1.f, 1.f, 63);

  std::vector<double> expected(elems);
  const double scale = 1.0 / std::sqrt(static_cast<double>(HEAD_DIM));
  std::vector<double> p(n);
  for (int b = 0; b < bh; ++b) {
    const size_t head = static_cast<size_t>(b) * n * HEAD_DIM;
    for (int i = 0; i < n; ++i) {
      double m = -INFINITY;
      for (int j = 0; j < n; ++j) {
        double s = 0.0;
        for (int c = 0; c < HEAD_DIM; ++c) {
          s += static_cast<double>(q[head + i * HEAD_DIM + c]) * k[head + j * HEAD_DIM + c];
        }
        p[j] = s * scale;
        m = std::fmax(m, p[j]);
      }
      double sum = 0.0;
      for (int j = 0; j < n; ++j) {
        p[j] = std::exp(p[j] - m);
        sum += p[j];
      }
      for (int c = 0; c < HEAD_DIM; ++c) {
        double acc = 0.0;
        for (int j = 0; j < n; ++j) acc += p[j] * v[head + j * HEAD_DIM + c];
        expected[head + i * HEAD_DIM + c] = acc / sum;
      }
    }
  }

  float *d_q, *d_k, *d_v, *d_o, *d_scratch;
  CUDA_CHECK(cudaMalloc(&d_q, elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_k, elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_v, elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_o, elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_scratch, static_cast<size_t>(bh) * n * n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_q, q.data(), elems * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, k.data(), elems * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, v.data(), elems * sizeof(float), cudaMemcpyHostToDevice));

  launch(d_q, d_k, d_v, d_o, bh, n, d_scratch);
  CUDA_CHECK_LAUNCH();
  std::vector<float> got(elems);
  CUDA_CHECK(cudaMemcpy(got.data(), d_o, elems * sizeof(float), cudaMemcpyDeviceToHost));
  const bool pass = lab::check_close("attention output", got, expected, 1e-4, 1e-5);

  const float ms = lab::time_ms([&] { launch(d_q, d_k, d_v, d_o, bh, n, d_scratch); });
  lab::report(name, ms, 4.0 * bh * n * static_cast<double>(n) * HEAD_DIM,
              4.0 * elems * sizeof(float));

  CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_o));
  CUDA_CHECK(cudaFree(d_scratch));
  return lab::finish(pass);
}
