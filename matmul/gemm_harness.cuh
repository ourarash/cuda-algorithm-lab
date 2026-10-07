/*
 * gemm_harness.cuh: shared host-side driver for the GEMM ladder.
 *
 * Every stage in matmul/ computes C = alpha * A @ B + beta * C for row-major
 * A (M x K), B (K x N), and C (M x N). The stages differ only in their kernel
 * and launch configuration, so this header owns everything else:
 * - random inputs from a fixed seed, with beta != 0 so the read-modify-write
 *   of C is exercised too;
 * - a CPU reference computed in double precision from exactly the values the
 *   GPU sees (for FP16 inputs, after rounding to half);
 * - validation with a relative tolerance, then warmed-up repeated timing and
 *   a GFLOP/s report;
 * - --quick for small sizes that are deliberately not multiples of the tile
 *   sizes (to exercise boundary handling), and --m/--n/--k overrides.
 */
#pragma once

#include <cuda_fp16.h>

#include <cstdio>
#include <type_traits>
#include <vector>

#include "lab.cuh"

struct GemmShape {
  int M, N, K;
};

// Some kernels only support sizes that are multiples of their tile or vector
// width. The harness refuses other sizes instead of producing wrong answers.
struct GemmRequirements {
  int m_multiple = 1;
  int n_multiple = 1;
  int k_multiple = 1;
};

template <typename T>
using GemmLauncher = void (*)(int M, int N, int K, float alpha, const T *A,
                              const T *B, float beta, float *C);

namespace gemm_detail {

template <typename T>
T from_float(float x);
template <>
inline float from_float<float>(float x) {
  return x;
}
template <>
inline __half from_float<__half>(float x) {
  return __float2half_rn(x);
}

inline double to_double(float x) { return x; }
inline double to_double(__half x) { return __half2float(x); }

// C = alpha * A @ B + beta * C0 in double precision. The i-k-j loop order
// streams through rows of B, which keeps the CPU reference reasonably fast.
inline std::vector<double> reference_gemm(int M, int N, int K, double alpha,
                                          const std::vector<double> &A,
                                          const std::vector<double> &B,
                                          double beta,
                                          const std::vector<float> &C0) {
  std::vector<double> C(static_cast<size_t>(M) * N);
  std::vector<double> row(N);
  for (int i = 0; i < M; ++i) {
    std::fill(row.begin(), row.end(), 0.0);
    for (int k = 0; k < K; ++k) {
      const double a = A[static_cast<size_t>(i) * K + k];
      const double *b = &B[static_cast<size_t>(k) * N];
      for (int j = 0; j < N; ++j) {
        row[j] += a * b[j];
      }
    }
    for (int j = 0; j < N; ++j) {
      const size_t idx = static_cast<size_t>(i) * N + j;
      C[idx] = alpha * row[j] + beta * C0[idx];
    }
  }
  return C;
}

}  // namespace gemm_detail

template <typename T>
int run_gemm(const char *name, int argc, char **argv, GemmShape full,
             GemmShape quick, GemmLauncher<T> launch,
             GemmRequirements req = {}) {
  lab::Args args(argc, argv);
  GemmShape s = args.quick() ? quick : full;
  s.M = static_cast<int>(args.get_int("m", s.M));
  s.N = static_cast<int>(args.get_int("n", s.N));
  s.K = static_cast<int>(args.get_int("k", s.K));
  const int M = s.M, N = s.N, K = s.K;

  lab::print_device();
  std::printf("%s: C = alpha * A @ B + beta * C with M=%d, N=%d, K=%d\n", name,
              M, N, K);
  if (M % req.m_multiple || N % req.n_multiple || K % req.k_multiple) {
    std::printf("This kernel requires M %% %d == 0, N %% %d == 0, K %% %d == 0\n",
                req.m_multiple, req.n_multiple, req.k_multiple);
    return lab::finish(false);
  }

  const float alpha = 1.0f;
  const float beta = 0.5f;
  const size_t a_size = static_cast<size_t>(M) * K;
  const size_t b_size = static_cast<size_t>(K) * N;
  const size_t c_size = static_cast<size_t>(M) * N;

  const std::vector<float> a32 = lab::random_uniform<float>(a_size, -1.f, 1.f, 11);
  const std::vector<float> b32 = lab::random_uniform<float>(b_size, -1.f, 1.f, 12);
  const std::vector<float> c0 = lab::random_uniform<float>(c_size, -1.f, 1.f, 13);

  std::vector<T> h_a(a_size), h_b(b_size);
  std::vector<double> a64(a_size), b64(b_size);
  for (size_t i = 0; i < a_size; ++i) {
    h_a[i] = gemm_detail::from_float<T>(a32[i]);
    a64[i] = gemm_detail::to_double(h_a[i]);
  }
  for (size_t i = 0; i < b_size; ++i) {
    h_b[i] = gemm_detail::from_float<T>(b32[i]);
    b64[i] = gemm_detail::to_double(h_b[i]);
  }

  T *d_a, *d_b;
  float *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, a_size * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_b, b_size * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_c, c_size * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), a_size * sizeof(T),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), b_size * sizeof(T),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_c, c0.data(), c_size * sizeof(float),
                        cudaMemcpyHostToDevice));

  // 1. One run from the initial C, validated against the CPU.
  launch(M, N, K, alpha, d_a, d_b, beta, d_c);
  CUDA_CHECK_LAUNCH();
  std::vector<float> h_c(c_size);
  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, c_size * sizeof(float),
                        cudaMemcpyDeviceToHost));

  const std::vector<double> expected =
      gemm_detail::reference_gemm(M, N, K, alpha, a64, b64, beta, c0);
  // FP32 accumulation over K terms of magnitude <= 1 stays well inside these
  // bounds; real indexing bugs produce errors of order 1. Tensor cores get a
  // looser bound because their internal accumulation order and rounding
  // differ from a sequential FP32 sum.
  const bool half_inputs = std::is_same<T, __half>::value;
  const bool pass = lab::check_close("C", h_c, expected,
                                     /*rtol=*/half_inputs ? 1e-3 : 1e-4,
                                     /*atol=*/half_inputs ? 1e-2 : 1e-3);

  // 2. Timing. Each run updates C in place, which does not affect speed.
  const float ms =
      lab::time_ms([&] { launch(M, N, K, alpha, d_a, d_b, beta, d_c); });
  lab::report(name, ms, 2.0 * M * N * K, /*bytes=*/0);

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  return lab::finish(pass);
}
