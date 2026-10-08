/*
 * 3D Stencil 0: Naive
 *
 * Intention:
 * One thread per grid point, reading its 7 inputs straight from global
 * memory. Consecutive threads (threadIdx.x) walk along x, so every one of the
 * 7 reads is coalesced.
 *
 * What is wasteful:
 * Every input value is needed by 7 different output points, so it is
 * requested 7 times. The L1 and L2 caches absorb part of that, but the
 * neighbours in z are a whole N x N plane apart and rarely still cached.
 */
#include "../stencil_harness.cuh"

__global__ void stencil_naive(const float *in, float *out, int n) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z;
  if (x < 1 || x >= n - 1 || y < 1 || y >= n - 1 || z < 1 || z >= n - 1) {
    return;
  }
  const size_t plane = static_cast<size_t>(n) * n;
  const size_t i = z * plane + static_cast<size_t>(y) * n + x;
  out[i] = C0 * in[i] + C1 * (in[i - 1] + in[i + 1] + in[i - n] + in[i + n] +
                              in[i - plane] + in[i + plane]);
}

void launch(const float *d_in, float *d_out, int n) {
  dim3 block(32, 8);
  dim3 grid(lab::ceil_div(n, 32), lab::ceil_div(n, 8), n);
  stencil_naive<<<grid, block>>>(d_in, d_out, n);
}

int main(int argc, char **argv) {
  return run_stencil("0. Naive", argc, argv, launch);
}
