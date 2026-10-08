/*
 * sort_harness.cuh: shared host-side driver for the key sorts (steps 02-04).
 *
 * Each step sorts n random 32-bit unsigned keys from d_in into d_out
 * (d_in is left untouched, so every timed run sorts the same unsorted data).
 * The result is checked exactly against std::sort, and the speed is reported
 * as time and as millions of keys per second.
 */
#pragma once

#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

#include "lab.cuh"

// Sorts d_in[0..n) into d_out. d_scratch holds scratch_bytes (at least 2n
// keys plus 4 MB).
using SortLauncher = void (*)(const unsigned int *d_in, unsigned int *d_out,
                              int n, void *d_scratch, size_t scratch_bytes);

inline int run_sort(const char *name, int argc, char **argv,
                    SortLauncher launch) {
  lab::Args args(argc, argv);
  const int n = static_cast<int>(
      args.get_int("n", args.quick() ? 100003 : 16000000));

  lab::print_device();
  std::printf("%s: %d random 32-bit keys\n", name, n);

  std::mt19937 gen(42);
  std::vector<unsigned int> input(n);
  for (auto &k : input) {
    k = gen();
  }
  std::vector<unsigned int> expected = input;
  std::sort(expected.begin(), expected.end());

  const size_t scratch_bytes = 2 * static_cast<size_t>(n) * sizeof(unsigned int) + (4 << 20);
  unsigned int *d_in, *d_out;
  void *d_scratch;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc(&d_scratch, scratch_bytes));
  CUDA_CHECK(cudaMemcpy(d_in, input.data(), n * sizeof(unsigned int),
                        cudaMemcpyHostToDevice));

  launch(d_in, d_out, n, d_scratch, scratch_bytes);
  CUDA_CHECK_LAUNCH();
  std::vector<unsigned int> got(n);
  CUDA_CHECK(cudaMemcpy(got.data(), d_out, n * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));
  const bool pass = lab::check_equal("sorted keys", got, expected);

  const float ms =
      lab::time_ms([&] { launch(d_in, d_out, n, d_scratch, scratch_bytes); });
  lab::report(name, ms, 0, 0);
  std::printf("%-30s %9.1f Mkeys/s\n", "Sort rate", n / (ms * 1e3));

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_scratch));
  return lab::finish(pass);
}
