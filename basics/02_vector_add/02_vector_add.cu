/*
 * Vector Add
 *
 * Intention:
 * This is the classic first CUDA program: add two vectors elementwise on the
 * GPU, check every element, and measure the achieved memory bandwidth.
 *
 * High-Level Algorithm:
 * - Allocate host and device buffers for A, B, and C.
 * - Copy A and B to the GPU.
 * - Launch one thread per element so each thread computes C[i] = A[i] + B[i].
 * - Copy C back, compare it with a CPU result, and report GB/s.
 *
 * Why GB/s:
 * Vector add does one addition per 12 bytes of memory traffic, so it is
 * limited by DRAM bandwidth, not arithmetic. Comparing the achieved GB/s with
 * the GPU's peak bandwidth tells you how close to optimal the kernel is.
 */
#include <cstdio>
#include <vector>

#include "lab.cuh"

__global__ void vector_add(const float *a, const float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

int main(int argc, char **argv) {
  lab::Args args(argc, argv);
  const int n = static_cast<int>(args.get_int("n", args.quick() ? 100003 : 1 << 24));
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);

  lab::print_device();
  printf("Vector add of %d elements\n", n);

  std::vector<float> h_a = lab::random_uniform<float>(n, -1.0f, 1.0f, 1);
  std::vector<float> h_b = lab::random_uniform<float>(n, -1.0f, 1.0f, 2);
  std::vector<float> h_c(n);

  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

  const int block_size = 256;
  // Ceiling division: enough blocks to cover every element.
  const int grid_size = lab::ceil_div(n, block_size);
  printf("Grid: %d blocks x %d threads\n", grid_size, block_size);

  vector_add<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
  CUDA_CHECK_LAUNCH();
  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

  std::vector<float> expected(n);
  for (int i = 0; i < n; ++i) {
    expected[i] = h_a[i] + h_b[i];
  }
  // Each GPU addition is a single IEEE float add, so the result is exact.
  const bool pass = lab::check_equal("c", h_c, expected);

  const float ms = lab::time_ms(
      [&] { vector_add<<<grid_size, block_size>>>(d_a, d_b, d_c, n); });
  lab::report("vector_add", ms, /*flops=*/n, /*bytes=*/3.0 * bytes);

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  return lab::finish(pass);
}
