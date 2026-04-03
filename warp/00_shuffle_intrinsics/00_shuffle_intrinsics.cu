/*
 * Warp Shuffle Intrinsics
 *
 * Intention:
 * This file demonstrates warp-level register exchange without shared memory.
 *
 * High-Level Algorithm:
 * - Launch exactly one warp.
 * - Give each lane a small integer value.
 * - Use __shfl_sync, __shfl_up_sync, and __shfl_xor_sync to exchange those
 *   values directly between lanes.
 * - Print the result seen by each lane.
 */
#include <cstdio>
#include <cuda_runtime.h>

__global__ void shfl_example_kernel() {
  int laneId = threadIdx.x % 32;
  int value = laneId;

  int bcast = __shfl_sync(0xFFFFFFFF, value, 0);    // Broadcast from lane 0.
  int up = __shfl_up_sync(0xFFFFFFFF, value, 1);    // Read from laneId - 1.
  int xorv = __shfl_xor_sync(0xFFFFFFFF, value, 1); // Read from laneId ^ 1.

  printf("Thread %d: val=%d, bcast=%d, up=%d, xor=%d\n", threadIdx.x, value,
         bcast, up, xorv);
}

int main() {
  shfl_example_kernel<<<1, 32>>>();
  cudaDeviceSynchronize();
  return 0;
}
