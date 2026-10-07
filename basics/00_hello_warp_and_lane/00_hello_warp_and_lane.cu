/*
 * Hello Warp And Lane
 *
 * Intention:
 * This tiny demo shows how CUDA threads are grouped into warps and how to
 * compute a thread's warp id and lane id from threadIdx.x.
 *
 * High-Level Algorithm:
 * - Launch a small block of threads.
 * - For each thread, compute:
 *   - warp_id = threadIdx.x / warpSize
 *   - lane_id = threadIdx.x % warpSize
 * - Print those ids so the mapping from threads to warps is visible.
 * - Ask the occupancy API how many such blocks fit on one SM.
 */
#include <cstdio>

#include "lab.cuh"

__global__ void hello_from_gpu() {
  int warp_id = threadIdx.x / warpSize;  // warpSize is 32 on all NVIDIA GPUs.
  int lane_id = threadIdx.x % warpSize;  // Position within the warp.
  printf("Thread %2d -> warp %d, lane %2d\n", threadIdx.x, warp_id, lane_id);
}

int main() {
  lab::print_device();

  const int block_size = 64;  // Two warps.

  // Occupancy: how many blocks of this kernel, at this block size, can be
  // resident on one SM at the same time. The answer depends on the kernel's
  // register and shared-memory use as well as the hardware limits.
  int blocks_per_sm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, hello_from_gpu, block_size,
      /*dynamicSMemSize=*/0));
  printf("Resident blocks per SM for this kernel at %d threads/block: %d\n",
         block_size, blocks_per_sm);

  hello_from_gpu<<<1, block_size>>>();
  CUDA_CHECK_LAUNCH();

  // Wait for the GPU to finish so its printf output is flushed.
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
