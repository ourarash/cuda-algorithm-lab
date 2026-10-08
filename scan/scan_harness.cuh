/*
 * scan_harness.cuh: shared host-side driver for the large-array scans
 * (steps 07-09).
 *
 * Each step computes an inclusive prefix sum of n ints on the GPU:
 * out[i] = in[0] + ... + in[i]. Integers make validation exact (no rounding
 * differences between summation orders), and integer scans are what stream
 * compaction and radix sort need (see compaction/ and sort/).
 *
 * The harness provides random input (values 0..3, so sums stay far below
 * 2^31), an exact check against a CPU scan, and warmed-up timing reported as
 * GB/s for the minimum traffic (read the input once, write the output once),
 * with % of peak DRAM bandwidth. --quick and --n set the size.
 */
#pragma once

#include <cstdio>
#include <random>
#include <vector>

#include "lab.cuh"

// Inclusive scan of d_in into d_out. d_scratch is zeroed once before the first
// call and holds scratch_bytes; a step that needs it clean on every call
// clears what it uses itself.
using ScanLauncher = void (*)(const int *d_in, int *d_out, int n,
                              void *d_scratch, size_t scratch_bytes);

inline int run_scan(const char *name, int argc, char **argv,
                    ScanLauncher launch) {
  lab::Args args(argc, argv);
  const int n = static_cast<int>(
      args.get_int("n", args.quick() ? 100003 : (1 << 24) + 123));

  lab::print_device();
  std::printf("%s: inclusive scan of %d ints\n", name, n);

  std::mt19937 gen(9);
  std::uniform_int_distribution<int> dist(0, 3);
  std::vector<int> input(n);
  for (int &v : input) {
    v = dist(gen);
  }
  std::vector<int> expected(n);
  int running = 0;
  for (int i = 0; i < n; ++i) {
    running += input[i];
    expected[i] = running;
  }

  const size_t scratch_bytes = static_cast<size_t>(n) * sizeof(int) + (1 << 20);
  int *d_in, *d_out;
  void *d_scratch;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_scratch, scratch_bytes));
  CUDA_CHECK(cudaMemset(d_scratch, 0, scratch_bytes));
  CUDA_CHECK(cudaMemcpy(d_in, input.data(), n * sizeof(int),
                        cudaMemcpyHostToDevice));

  launch(d_in, d_out, n, d_scratch, scratch_bytes);
  CUDA_CHECK_LAUNCH();
  std::vector<int> got(n);
  CUDA_CHECK(cudaMemcpy(got.data(), d_out, n * sizeof(int),
                        cudaMemcpyDeviceToHost));
  const bool pass = lab::check_equal("inclusive scan", got, expected);

  const float ms = lab::time_ms(
      [&] { launch(d_in, d_out, n, d_scratch, scratch_bytes); });
  lab::report(name, ms, /*flops=*/n, /*bytes=*/2.0 * n * sizeof(int));

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_scratch));
  return lab::finish(pass);
}
