/*
 * lab.cuh: shared helpers for every example in CUDA Algorithm Lab.
 *
 * The examples include this header so each file can stay focused on its
 * kernels. It provides:
 * - CUDA_CHECK: abort with the file and line of any failing CUDA runtime call.
 * - lab::Args: a tiny command-line parser (--quick, --n=4096, --n 4096).
 * - lab::time_ms: warmup plus repeated timing with CUDA events (median).
 * - lab::report: time, GFLOP/s, GB/s, and % of peak DRAM bandwidth.
 * - lab::check_close / lab::check_equal: validation that prints mismatches.
 * - lab::finish: prints PASS or FAIL and returns the process exit code, so
 *   ctest and CI can detect a wrong answer.
 */
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                  \
  do {                                                                    \
    cudaError_t err_ = (call);                                            \
    if (err_ != cudaSuccess) {                                            \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                \
                   cudaGetErrorName(err_), __FILE__, __LINE__,            \
                   cudaGetErrorString(err_));                             \
      std::exit(EXIT_FAILURE);                                            \
    }                                                                     \
  } while (0)

// Kernel launches do not return an error code. Invalid launch configurations
// (for example, too many threads per block) are reported here instead.
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())

namespace lab {

template <typename T>
__host__ __device__ constexpr T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------
class Args {
 public:
  Args(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
      args_.emplace_back(argv[i]);
    }
  }

  bool has(const std::string& flag) const {
    return std::find(args_.begin(), args_.end(), flag) != args_.end();
  }

  // --quick shrinks problem sizes so sanitizer runs and smoke tests finish
  // quickly. ctest passes it to the compute-sanitizer tests.
  bool quick() const { return has("--quick"); }

  // Reads "--name=value" or "--name value"; returns fallback when absent.
  long long get_int(const std::string& name, long long fallback) const {
    const std::string key = "--" + name;
    for (size_t i = 0; i < args_.size(); ++i) {
      if (args_[i] == key && i + 1 < args_.size()) {
        return std::atoll(args_[i + 1].c_str());
      }
      if (args_[i].rfind(key + "=", 0) == 0) {
        return std::atoll(args_[i].c_str() + key.size() + 1);
      }
    }
    return fallback;
  }

 private:
  std::vector<std::string> args_;
};

// ---------------------------------------------------------------------------
// Device information
// ---------------------------------------------------------------------------

// Theoretical DRAM bandwidth in GB/s: two transfers per memory clock (DDR)
// times the bus width in bytes. Returns 0 if the driver does not report it.
inline double peak_bandwidth_gbs() {
  int device = 0;
  int mem_clock_khz = 0;
  int bus_width_bits = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate,
                                    device));
  CUDA_CHECK(cudaDeviceGetAttribute(&bus_width_bits,
                                    cudaDevAttrGlobalMemoryBusWidth, device));
  return 2.0 * mem_clock_khz * 1e3 * (bus_width_bits / 8.0) / 1e9;
}

inline void print_device() {
  int device = 0;
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  std::printf("GPU: %s (sm_%d%d, %d SMs, %.0f GB/s peak DRAM bandwidth)\n",
              prop.name, prop.major, prop.minor, prop.multiProcessorCount,
              peak_bandwidth_gbs());
}

// ---------------------------------------------------------------------------
// Timing and throughput
// ---------------------------------------------------------------------------

// Times `launch`, a callable that enqueues GPU work on the default stream.
// The untimed warmup runs absorb one-time costs such as lazy module loading,
// which would otherwise dominate a single cold measurement. Returns the median
// of `reps` timed runs in milliseconds.
template <typename Launch>
float time_ms(Launch&& launch, int warmup = 3, int reps = 20) {
  for (int i = 0; i < warmup; ++i) {
    launch();
  }
  CUDA_CHECK_LAUNCH();

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<float> times(reps);
  for (int i = 0; i < reps; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
  }
  CUDA_CHECK_LAUNCH();
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  std::nth_element(times.begin(), times.begin() + reps / 2, times.end());
  return times[reps / 2];
}

// Prints one performance line. Pass 0 for `flops` or `bytes` to omit that
// column. `bytes` should be the minimum traffic the algorithm needs (for
// example, read the input once), which gives the "effective bandwidth".
inline void report(const char* label, float ms, double flops, double bytes) {
  std::printf("%-30s %9.3f ms", label, ms);
  if (flops > 0) {
    std::printf(" | %9.1f GFLOP/s", flops / (ms * 1e6));
  }
  if (bytes > 0) {
    const double gbs = bytes / (ms * 1e6);
    std::printf(" | %8.1f GB/s", gbs);
    const double peak = peak_bandwidth_gbs();
    if (peak > 0) {
      std::printf(" (%.0f%% of peak)", 100.0 * gbs / peak);
    }
  }
  std::printf("\n");
}

// ---------------------------------------------------------------------------
// Test data and validation
// ---------------------------------------------------------------------------

// Uniform random values from a fixed seed, so every run sees the same input.
template <typename T>
std::vector<T> random_uniform(size_t n, T lo, T hi, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<T> dist(lo, hi);
  std::vector<T> values(n);
  for (T& v : values) {
    v = dist(gen);
  }
  return values;
}

// Element-wise check: |got - expected| <= atol + rtol * |expected|.
// `expected` is usually computed on the CPU in double precision, so it is
// more accurate than the GPU result and the tolerance only has to absorb the
// GPU's rounding. NaN in `got` always fails.
template <typename T, typename R>
bool check_close(const char* what, const T* got, const R* expected, size_t n,
                 double rtol, double atol) {
  size_t mismatches = 0;
  double max_err = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double g = static_cast<double>(got[i]);
    const double e = static_cast<double>(expected[i]);
    const double err = std::fabs(g - e);
    if (err > max_err) {
      max_err = err;
    }
    if (!(err <= atol + rtol * std::fabs(e))) {
      if (mismatches < 5) {
        std::fprintf(stderr, "  %s[%zu]: got %.9g, expected %.9g\n", what, i,
                     g, e);
      }
      ++mismatches;
    }
  }
  std::printf("Check %-24s %s (max abs error %.3g", what,
              mismatches == 0 ? "ok" : "FAILED", max_err);
  if (mismatches > 0) {
    std::printf(", %zu of %zu mismatched", mismatches, n);
  }
  std::printf(")\n");
  return mismatches == 0;
}

template <typename T, typename R>
bool check_close(const char* what, const std::vector<T>& got,
                 const std::vector<R>& expected, double rtol, double atol) {
  if (got.size() != expected.size()) {
    std::printf("Check %-24s FAILED (size %zu, expected %zu)\n", what,
                got.size(), expected.size());
    return false;
  }
  return check_close(what, got.data(), expected.data(), got.size(), rtol,
                     atol);
}

// Exact comparison, for integer results and permutations.
template <typename T>
bool check_equal(const char* what, const std::vector<T>& got,
                 const std::vector<T>& expected) {
  if (got.size() != expected.size()) {
    std::printf("Check %-24s FAILED (size %zu, expected %zu)\n", what,
                got.size(), expected.size());
    return false;
  }
  size_t mismatches = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    if (!(got[i] == expected[i])) {
      if (mismatches < 5) {
        std::fprintf(stderr, "  %s[%zu]: got %.9g, expected %.9g\n", what, i,
                     static_cast<double>(got[i]),
                     static_cast<double>(expected[i]));
      }
      ++mismatches;
    }
  }
  std::printf("Check %-24s %s", what, mismatches == 0 ? "ok" : "FAILED");
  if (mismatches > 0) {
    std::printf(" (%zu of %zu mismatched)", mismatches, got.size());
  }
  std::printf("\n");
  return mismatches == 0;
}

// Prints the verdict and returns the process exit code.
inline int finish(bool pass) {
  std::printf("%s\n", pass ? "PASS" : "FAIL");
  return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace lab
