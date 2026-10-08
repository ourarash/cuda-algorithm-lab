/*
 * Occupancy and __launch_bounds__
 *
 * Intention:
 * Occupancy is the number of warps resident on an SM compared with the
 * maximum it supports. More resident warps give the scheduler more work to
 * switch to while others wait on memory. Occupancy is limited by whichever
 * per-SM resource runs out first: registers, shared memory, threads, or
 * blocks. This example shows the register limit and the knob that controls
 * it.
 *
 * High-Level Algorithm:
 * - work(): each thread keeps 32 running values in registers, so the
 *   compiler needs many registers per thread.
 * - work_bounded(): the same code with __launch_bounds__(256, 8), which tells
 *   the compiler "256 threads per block, and I want 8 blocks per SM". To fit,
 *   it must cap registers per thread and spill the rest to local memory.
 * - For both, print registers per thread (cudaFuncGetAttributes), resident
 *   blocks per SM and occupancy (cudaOccupancyMaxActiveBlocksPerMultiprocessor),
 *   the block size the runtime suggests (cudaOccupancyMaxPotentialBlockSize),
 *   and the measured time. Both must produce identical results.
 *
 * Higher occupancy is not automatically faster: if forcing it causes spills,
 * the extra local-memory traffic can cost more than the latency hiding gains.
 * Measure.
 */
#include <vector>

#include "lab.cuh"

constexpr int THREADS = 256;
constexpr int VALUES = 32;

template <int Dummy>
__device__ __forceinline__ float body(const float *in, int i, int n) {
  float v[VALUES];
#pragma unroll
  for (int k = 0; k < VALUES; ++k) v[k] = in[(i + k * 997) % n];
  for (int iter = 0; iter < 32; ++iter) {
#pragma unroll
    for (int k = 0; k < VALUES; ++k) v[k] = v[k] * 0.999f + v[(k + 1) % VALUES] * 0.001f;
  }
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < VALUES; ++k) sum += v[k];
  return sum;
}

__global__ void work(const float *in, float *out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = body<0>(in, i, n);
}

__global__ void __launch_bounds__(THREADS, 8)
    work_bounded(const float *in, float *out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = body<1>(in, i, n);
}

template <typename Kernel>
void describe(const char *name, Kernel kernel) {
  cudaFuncAttributes attr;
  CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
  int blocks_per_sm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, THREADS, 0));
  int min_grid = 0, suggested_block = 0;
  CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid, &suggested_block, kernel, 0, 0));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  const double occupancy = 100.0 * blocks_per_sm * THREADS / prop.maxThreadsPerMultiProcessor;
  printf("%-14s %3d registers/thread, %4zu B local (spill) memory, %d blocks/SM of %d "
         "threads = %3.0f%% occupancy; suggested block size %d\n",
         name, attr.numRegs, attr.localSizeBytes, blocks_per_sm, THREADS, occupancy,
         suggested_block);
}

int main(int argc, char **argv) {
  lab::Args args(argc, argv);
  const int n = static_cast<int>(args.get_int("n", args.quick() ? 100003 : 1 << 22));

  lab::print_device();
  describe("work", work);
  describe("work_bounded", work_bounded);

  const std::vector<float> h_in = lab::random_uniform<float>(n, 0.f, 1.f, 81);
  float *d_in, *d_a, *d_b;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  const int blocks = lab::ceil_div(n, THREADS);

  work<<<blocks, THREADS>>>(d_in, d_a, n);
  work_bounded<<<blocks, THREADS>>>(d_in, d_b, n);
  CUDA_CHECK_LAUNCH();
  std::vector<float> a(n), b(n);
  CUDA_CHECK(cudaMemcpy(a.data(), d_a, n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(b.data(), d_b, n * sizeof(float), cudaMemcpyDeviceToHost));
  // Register allocation does not change the arithmetic, so results match.
  const bool pass = lab::check_close("bounded == unbounded", b, a, 1e-6, 1e-7);

  lab::report("work", lab::time_ms([&] { work<<<blocks, THREADS>>>(d_in, d_a, n); }), 0, 0);
  lab::report("work_bounded", lab::time_ms([&] { work_bounded<<<blocks, THREADS>>>(d_in, d_b, n); }), 0, 0);

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  return lab::finish(pass);
}
