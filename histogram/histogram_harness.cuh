/*
 * histogram_harness.cuh: shared host-side driver for the histogram ladder.
 *
 * Every step counts how often each byte value occurs in an array of n bytes:
 * 256 bins of unsigned int. A step provides one launcher; the harness owns
 * the rest. It runs two inputs, because the best technique depends on the
 * data:
 * - uniform: random bytes, so the 256 bins are hit about equally and atomic
 *   updates rarely collide;
 * - skewed: image-like data with long runs of the same value and a few
 *   dominant values, so many threads update the same bins at once
 *   (contention), and neighbouring elements are often equal.
 * Each input is validated exactly against a CPU histogram and timed; the
 * histogram is cleared inside each timed run. Bandwidth counts each input
 * byte once. --quick and --n set the size.
 */
#pragma once

#include <cstdio>
#include <random>
#include <vector>

#include "lab.cuh"

constexpr int NUM_BINS = 256;

// Adds the counts of d_in[0..n) into d_hist (which arrives zeroed).
// d_scratch holds scratch_bytes of temporary storage (used by CUB).
using HistogramLauncher = void (*)(const unsigned char *d_in, int n,
                                   unsigned int *d_hist, void *d_scratch,
                                   size_t scratch_bytes);

inline std::vector<unsigned char> uniform_bytes(int n) {
  std::mt19937 gen(21);
  std::uniform_int_distribution<int> dist(0, 255);
  std::vector<unsigned char> v(n);
  for (auto &x : v) {
    x = static_cast<unsigned char>(dist(gen));
  }
  return v;
}

// Runs of 1-64 equal values; 80% of runs use one of four "background" values.
inline std::vector<unsigned char> skewed_bytes(int n) {
  std::mt19937 gen(22);
  std::uniform_int_distribution<int> run(1, 64), any(0, 255), common(0, 3),
      coin(0, 9);
  std::vector<unsigned char> v(n);
  for (int i = 0; i < n;) {
    const int value = coin(gen) < 8 ? 100 + common(gen) : any(gen);
    for (int r = run(gen); r > 0 && i < n; --r) {
      v[i++] = static_cast<unsigned char>(value);
    }
  }
  return v;
}

inline bool run_one(const char *name, const char *data_name,
                    const std::vector<unsigned char> &input,
                    HistogramLauncher launch, unsigned char *d_in,
                    unsigned int *d_hist, void *d_scratch,
                    size_t scratch_bytes) {
  const int n = static_cast<int>(input.size());
  CUDA_CHECK(cudaMemcpy(d_in, input.data(), n, cudaMemcpyHostToDevice));

  std::vector<unsigned int> expected(NUM_BINS, 0);
  for (unsigned char x : input) {
    ++expected[x];
  }

  CUDA_CHECK(cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int)));
  launch(d_in, n, d_hist, d_scratch, scratch_bytes);
  CUDA_CHECK_LAUNCH();
  std::vector<unsigned int> got(NUM_BINS);
  CUDA_CHECK(cudaMemcpy(got.data(), d_hist, NUM_BINS * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));
  char label[64];
  std::snprintf(label, sizeof(label), "%s histogram", data_name);
  const bool pass = lab::check_equal(label, got, expected);

  const float ms = lab::time_ms([&] {
    CUDA_CHECK(cudaMemsetAsync(d_hist, 0, NUM_BINS * sizeof(unsigned int)));
    launch(d_in, n, d_hist, d_scratch, scratch_bytes);
  });
  std::snprintf(label, sizeof(label), "%s (%s)", name, data_name);
  lab::report(label, ms, 0, static_cast<double>(n));
  return pass;
}

inline int run_histogram(const char *name, int argc, char **argv,
                         HistogramLauncher launch) {
  lab::Args args(argc, argv);
  const int n = static_cast<int>(
      args.get_int("n", args.quick() ? 100003 : (1 << 26) + 123));

  lab::print_device();
  std::printf("%s: %d-bin histogram of %d bytes\n", name, NUM_BINS, n);

  const size_t scratch_bytes = 1 << 20;
  unsigned char *d_in;
  unsigned int *d_hist;
  void *d_scratch;
  CUDA_CHECK(cudaMalloc(&d_in, n));
  CUDA_CHECK(cudaMalloc(&d_hist, NUM_BINS * sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc(&d_scratch, scratch_bytes));

  bool pass = run_one(name, "uniform", uniform_bytes(n), launch, d_in, d_hist,
                      d_scratch, scratch_bytes);
  pass = run_one(name, "skewed", skewed_bytes(n), launch, d_in, d_hist,
                 d_scratch, scratch_bytes) && pass;

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_hist));
  CUDA_CHECK(cudaFree(d_scratch));
  return lab::finish(pass);
}

// Grid size for grid-stride kernels: enough blocks to fill every SM.
inline int histogram_blocks(int work_items, int threads) {
  int device = 0;
  int sms = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  const int needed = lab::ceil_div(work_items, threads);
  return needed < sms * 8 ? needed : sms * 8;
}
