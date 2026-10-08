/*
 * compaction_harness.cuh: shared host-side driver for stream compaction.
 *
 * Stream compaction (also called filtering or select) copies the elements
 * that satisfy a predicate into a dense output array and reports how many
 * there were. Here: keep the ints below KEEP_BELOW, about 30% of the input.
 * It is a building block for many GPU algorithms (removing dead particles,
 * collecting active work items, sparse matrix construction, ...).
 *
 * A *stable* compaction keeps the selected elements in their input order and
 * is checked exactly. An unstable one (step 01) may emit them in any order;
 * it is checked as a multiset (same elements, same count). Bandwidth counts
 * reading the input once and writing the selected elements once.
 */
#pragma once

#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

#include "lab.cuh"

constexpr int KEEP_BELOW = 300;  // Inputs are uniform in [0, 1000)

__host__ __device__ __forceinline__ bool keep(int x) { return x < KEEP_BELOW; }

// Writes the selected elements of d_in to d_out[0 .. count) and count to
// *d_count. d_scratch holds scratch_bytes of temporary storage.
using CompactLauncher = void (*)(const int *d_in, int *d_out, int *d_count,
                                 int n, void *d_scratch, size_t scratch_bytes);

inline int run_compaction(const char *name, int argc, char **argv,
                          CompactLauncher launch, bool stable) {
  lab::Args args(argc, argv);
  const int n = static_cast<int>(
      args.get_int("n", args.quick() ? 100003 : (1 << 24) + 123));

  lab::print_device();
  std::printf("%s: keep x < %d from %d ints (%s)\n", name, KEEP_BELOW, n,
              stable ? "stable" : "unstable");

  std::mt19937 gen(13);
  std::uniform_int_distribution<int> dist(0, 999);
  std::vector<int> input(n);
  std::vector<int> expected;
  for (int &v : input) {
    v = dist(gen);
    if (keep(v)) {
      expected.push_back(v);
    }
  }

  const size_t scratch_bytes = static_cast<size_t>(n) * sizeof(int) + (1 << 20);
  int *d_in, *d_out, *d_count;
  void *d_scratch;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_scratch, scratch_bytes));
  CUDA_CHECK(cudaMemcpy(d_in, input.data(), n * sizeof(int),
                        cudaMemcpyHostToDevice));

  launch(d_in, d_out, d_count, n, d_scratch, scratch_bytes);
  CUDA_CHECK_LAUNCH();
  int count = 0;
  CUDA_CHECK(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
  std::printf("Selected %d of %d elements\n", count, n);
  bool pass = count == static_cast<int>(expected.size());
  if (!pass) {
    std::printf("Check count FAILED (got %d, expected %zu)\n", count,
                expected.size());
  } else {
    std::vector<int> got(count);
    CUDA_CHECK(cudaMemcpy(got.data(), d_out, count * sizeof(int),
                          cudaMemcpyDeviceToHost));
    if (!stable) {
      std::sort(got.begin(), got.end());
      std::sort(expected.begin(), expected.end());
    }
    pass = lab::check_equal(stable ? "selected elements (in order)"
                                   : "selected elements (as a set)",
                            got, expected);
  }

  const float ms = lab::time_ms(
      [&] { launch(d_in, d_out, d_count, n, d_scratch, scratch_bytes); });
  lab::report(name, ms, 0,
              (static_cast<double>(n) + expected.size()) * sizeof(int));

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_count));
  CUDA_CHECK(cudaFree(d_scratch));
  return lab::finish(pass);
}
