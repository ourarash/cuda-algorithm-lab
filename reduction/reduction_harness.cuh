/*
 * reduction_harness.cuh: shared host-side driver for the reduction ladder.
 *
 * Every stage in reduction/ computes the sum of n floats entirely on the GPU
 * and writes it to a single float in device memory. A stage provides one
 * launcher function; this header owns everything else:
 * - Exact check: about a million small integers (0..3) at an awkward size.
 *   Every partial sum is an integer below 2^24, which a float represents
 *   exactly, so the GPU result must match exactly no matter in which order it
 *   adds. A single lost or double-counted element fails this check.
 * - Accuracy check: millions of random floats in [0, 1) against a
 *   double-precision CPU sum, with a relative tolerance.
 * - Timing: warmed-up, repeated runs; reported as GB/s of input read (each
 *   element once), with % of peak DRAM bandwidth. Reduction does one add per
 *   4 bytes, so it is bandwidth-bound and GB/s is the number that matters.
 * - --quick for a small size and --n to choose the size.
 */
#pragma once

#include <cstdio>
#include <random>
#include <vector>

#include "lab.cuh"

// Sums n floats from d_in into *d_out. d_scratch holds at least n + 1024
// floats for partial results. d_in may be overwritten (00_naive reduces in
// place).
using ReduceLauncher = void (*)(float *d_in, int n, float *d_out,
                                float *d_scratch);

// Applies a block-level reduction pass repeatedly until one value remains:
// n inputs -> ceil(n / elems_per_block) partial sums -> ... -> 1. `pass(src,
// dst, n, blocks)` launches one pass. Partial sums ping-pong between two
// regions of the scratch buffer.
template <typename Pass>
void reduce_in_passes(float *d_in, int n, float *d_out, float *d_scratch,
                      int elems_per_block, Pass pass) {
  float *buffers[2] = {d_scratch,
                       d_scratch + lab::ceil_div(n, elems_per_block)};
  float *src = d_in;
  for (int level = 0;; ++level) {
    const int blocks = lab::ceil_div(n, elems_per_block);
    float *dst = blocks == 1 ? d_out : buffers[level % 2];
    pass(src, dst, n, blocks);
    if (blocks == 1) {
      return;
    }
    src = dst;
    n = blocks;
  }
}

// Grid size for grid-stride kernels: enough blocks to fill every SM (8
// resident blocks of 256 threads each), but no more than the data needs.
inline int grid_stride_blocks(int n, int elems_per_block) {
  static int max_blocks = 0;
  if (max_blocks == 0) {
    int device = 0;
    int sms = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    max_blocks = sms * 8;
  }
  const int needed = lab::ceil_div(n, elems_per_block);
  return needed < max_blocks ? needed : max_blocks;
}

inline float run_once(ReduceLauncher launch, const std::vector<float> &input,
                      float *d_in, float *d_out, float *d_scratch) {
  const int n = static_cast<int>(input.size());
  CUDA_CHECK(cudaMemcpy(d_in, input.data(), n * sizeof(float),
                        cudaMemcpyHostToDevice));
  launch(d_in, n, d_out, d_scratch);
  CUDA_CHECK_LAUNCH();
  float result = 0.0f;
  CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost));
  return result;
}

inline int run_reduction(const char *name, int argc, char **argv,
                         ReduceLauncher launch) {
  lab::Args args(argc, argv);
  const int n = static_cast<int>(
      args.get_int("n", args.quick() ? 100003 : (1 << 24) + 123));
  const int n_exact = n < 1000003 ? n : 1000003;

  lab::print_device();
  std::printf("%s: sum of %d floats\n", name, n);

  float *d_in, *d_out, *d_scratch;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_scratch, (static_cast<size_t>(n) + 1024) * sizeof(float)));

  // 1. Exact check with small integers.
  std::mt19937 gen(5);
  std::uniform_int_distribution<int> small(0, 3);
  std::vector<float> integers(n_exact);
  double integer_sum = 0.0;
  for (float &v : integers) {
    v = static_cast<float>(small(gen));
    integer_sum += v;
  }
  const std::vector<float> got_exact = {
      run_once(launch, integers, d_in, d_out, d_scratch)};
  const std::vector<double> want_exact = {integer_sum};
  bool pass = lab::check_close("exact integer sum", got_exact, want_exact, 0.0, 0.0);

  // 2. Accuracy check with random floats.
  const std::vector<float> input = lab::random_uniform<float>(n, 0.f, 1.f, 7);
  double sum = 0.0;
  for (float v : input) {
    sum += v;
  }
  const std::vector<float> got = {run_once(launch, input, d_in, d_out, d_scratch)};
  const std::vector<double> want = {sum};
  pass = lab::check_close("random float sum", got, want, 1e-5, 0.0) && pass;

  // 3. Timing (00_naive reduces in place, which changes the values but not
  // the memory traffic).
  const float ms = lab::time_ms([&] { launch(d_in, n, d_out, d_scratch); });
  lab::report(name, ms, /*flops=*/n, /*bytes=*/static_cast<double>(n) * sizeof(float));

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_scratch));
  return lab::finish(pass);
}
