/*
 * Warp Shuffle Intrinsics
 *
 * Intention:
 * This file demonstrates warp-level register exchange without shared memory.
 *
 * High-Level Algorithm:
 * - Launch exactly one warp.
 * - Give each lane a small integer value (its lane id).
 * - Use __shfl_sync, __shfl_up_sync, and __shfl_xor_sync to exchange those
 *   values directly between lanes' registers.
 * - Print what each lane received and check it against the documented
 *   semantics on the CPU.
 *
 * The 0xFFFFFFFF mask says all 32 lanes participate. Every lane named in the
 * mask must execute the same shuffle, or the behavior is undefined.
 */
#include <cstdio>
#include <vector>

#include "lab.cuh"

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xFFFFFFFFu;

__global__ void shfl_example_kernel(int *bcast_out, int *up_out, int *xor_out) {
  int lane = threadIdx.x % kWarpSize;
  int value = lane;

  // Broadcast: every lane reads lane 0's value.
  int bcast = __shfl_sync(kFullMask, value, 0);
  // Shift up: lane i reads lane i - 1. Lane 0 has no source lane, so it keeps
  // its own value.
  int up = __shfl_up_sync(kFullMask, value, 1);
  // Butterfly: lane i reads lane i ^ 1 (pairs 0<->1, 2<->3, ...). XOR
  // shuffles with offsets 16, 8, 4, 2, 1 are the core of a warp reduction.
  int xorv = __shfl_xor_sync(kFullMask, value, 1);

  printf("Lane %2d: value=%2d bcast=%2d up=%2d xor=%2d\n", lane, value, bcast,
         up, xorv);
  bcast_out[lane] = bcast;
  up_out[lane] = up;
  xor_out[lane] = xorv;
}

int main() {
  int *d_out;
  CUDA_CHECK(cudaMalloc(&d_out, 3 * kWarpSize * sizeof(int)));

  shfl_example_kernel<<<1, kWarpSize>>>(d_out, d_out + kWarpSize,
                                        d_out + 2 * kWarpSize);
  CUDA_CHECK_LAUNCH();

  std::vector<int> h_out(3 * kWarpSize);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(int),
                        cudaMemcpyDeviceToHost));

  std::vector<int> expected(3 * kWarpSize);
  for (int lane = 0; lane < kWarpSize; ++lane) {
    expected[lane] = 0;
    expected[kWarpSize + lane] = lane == 0 ? 0 : lane - 1;
    expected[2 * kWarpSize + lane] = lane ^ 1;
  }
  const bool pass = lab::check_equal("shuffle results", h_out, expected);

  CUDA_CHECK(cudaFree(d_out));
  return lab::finish(pass);
}
