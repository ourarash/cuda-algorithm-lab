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
 * - the same problem run through cuBLAS as the baseline: cublasSgemm for FP32
 *   stages, and cublasGemmEx with FP16 inputs and FP32 accumulation for the
 *   Tensor Core stages, so the comparison is always like for like;
 * - --quick for small sizes that are deliberately not multiples of the tile
 *   sizes (to exercise boundary handling), and --m/--n/--k overrides.
 */
#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <type_traits>
#include <vector>

#include "lab.cuh"

#define CUBLAS_CHECK(call)                                                 \
  do {                                                                     \
    cublasStatus_t status_ = (call);                                       \
    if (status_ != CUBLAS_STATUS_SUCCESS) {                                \
      std::fprintf(stderr, "cuBLAS error %d at %s:%d\n",                   \
                   static_cast<int>(status_), __FILE__, __LINE__);         \
      std::exit(EXIT_FAILURE);                                             \
    }                                                                      \
  } while (0)

struct GemmShape {
  int M, N, K;
};

// What a kernel needs from the problem and the hardware. The harness refuses
// unsupported sizes instead of producing wrong answers, and skips (exit code
// 77) on GPUs the kernel cannot run on.
struct GemmRequirements {
  int m_multiple = 1;
  int n_multiple = 1;
  int k_multiple = 1;
  // Minimum compute capability as one number (80 = 8.0), or 0 for any.
  int min_compute_capability = 0;
  // Exact compute capability, for architecture-specific features such as
  // Hopper's wgmma (sm_90a code only runs on compute capability 9.0).
  int exact_compute_capability = 0;
  // Pass B to the kernel as an N x K row-major array (B transposed, also
  // called "K-major" or column-major B). Tensor Core instructions on Hopper
  // read both operands K-major. The reference and cuBLAS still use B.
  bool b_k_major = false;
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

// Row-major C = A * B through cuBLAS, which is column-major: computing
// C^T = B^T * A^T reinterprets every row-major matrix as its column-major
// transpose, so no data has to move (see libraries/00_cublas_gemm).
inline void cublas_gemm(cublasHandle_t handle, int M, int N, int K,
                        const float *A, const float *B, float *C) {
  const float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                           B, N, A, K, &beta, C, N));
}

inline void cublas_gemm(cublasHandle_t handle, int M, int N, int K,
                        const __half *A, const __half *B, float *C) {
  const float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                            B, CUDA_R_16F, N, A, CUDA_R_16F, K, &beta, C,
                            CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT));
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

  const int cc = lab::compute_capability();
  if (req.exact_compute_capability && cc != req.exact_compute_capability) {
    char reason[128];
    std::snprintf(reason, sizeof(reason),
                  "this kernel needs compute capability %d.%d exactly",
                  req.exact_compute_capability / 10,
                  req.exact_compute_capability % 10);
    return lab::skip(reason);
  }
  if (cc < req.min_compute_capability) {
    char reason[128];
    std::snprintf(reason, sizeof(reason),
                  "this kernel needs compute capability %d.%d or newer",
                  req.min_compute_capability / 10,
                  req.min_compute_capability % 10);
    return lab::skip(reason);
  }
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

  std::vector<T> h_a(a_size), h_b(b_size), h_b_kernel(b_size);
  std::vector<double> a64(a_size), b64(b_size);
  for (size_t i = 0; i < a_size; ++i) {
    h_a[i] = gemm_detail::from_float<T>(a32[i]);
    a64[i] = gemm_detail::to_double(h_a[i]);
  }
  for (size_t i = 0; i < b_size; ++i) {
    h_b[i] = gemm_detail::from_float<T>(b32[i]);
    b64[i] = gemm_detail::to_double(h_b[i]);
  }
  // The kernel's copy of B, transposed to N x K when it reads B K-major.
  for (int k = 0; k < K; ++k) {
    for (int n = 0; n < N; ++n) {
      const size_t src = static_cast<size_t>(k) * N + n;
      const size_t dst =
          req.b_k_major ? static_cast<size_t>(n) * K + k : src;
      h_b_kernel[dst] = h_b[src];
    }
  }

  T *d_a, *d_b, *d_b_kernel;
  float *d_c, *d_c_cublas;
  CUDA_CHECK(cudaMalloc(&d_a, a_size * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_b, b_size * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_b_kernel, b_size * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_c, c_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c_cublas, c_size * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), a_size * sizeof(T),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), b_size * sizeof(T),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b_kernel, h_b_kernel.data(), b_size * sizeof(T),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_c, c0.data(), c_size * sizeof(float),
                        cudaMemcpyHostToDevice));

  // 1. One run from the initial C, validated against the CPU.
  launch(M, N, K, alpha, d_a, d_b_kernel, beta, d_c);
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
  const double flops = 2.0 * M * N * K;
  const float ms =
      lab::time_ms([&] { launch(M, N, K, alpha, d_a, d_b_kernel, beta, d_c); });
  lab::report(name, ms, flops, /*bytes=*/0);

  // 3. The same problem through cuBLAS, as the baseline to beat.
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  const float cublas_ms = lab::time_ms([&] {
    gemm_detail::cublas_gemm(handle, M, N, K, d_a, d_b, d_c_cublas);
  });
  lab::report(half_inputs ? "cuBLAS (cublasGemmEx, FP16)" : "cuBLAS (cublasSgemm)",
              cublas_ms, flops, /*bytes=*/0);
  std::printf("%-30s %9.1f %%\n", "Speed relative to cuBLAS",
              100.0 * cublas_ms / ms);
  CUBLAS_CHECK(cublasDestroy(handle));

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_b_kernel));
  CUDA_CHECK(cudaFree(d_c));
  CUDA_CHECK(cudaFree(d_c_cublas));
  return lab::finish(pass);
}
