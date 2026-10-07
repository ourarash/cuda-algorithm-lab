/*
 * Hello Thread Hierarchy
 *
 * Intention:
 * This file is a minimal demonstration of CUDA's grid / block / thread
 * hierarchy with a 3D thread block.
 *
 * High-Level Algorithm:
 * - Launch a grid of two blocks, each a 3 x 4 x 2 block of threads.
 * - Each thread prints its block index and its local (x, y, z) coordinates.
 * - The output makes the hierarchy visible: thread coordinates repeat in every
 *   block, and only blockIdx tells the two blocks apart.
 */
#include <cstdio>

#include "lab.cuh"

__global__ void hello_from_gpu_3d() {
  printf("Block %d: thread (%d, %d, %d)\n", blockIdx.x, threadIdx.x,
         threadIdx.y, threadIdx.z);
}

int main() {
  dim3 threads_per_block(3, 4, 2);  // 24 threads spread across x, y, and z.
  dim3 num_blocks(2);               // 2 blocks in x.

  hello_from_gpu_3d<<<num_blocks, threads_per_block>>>();
  CUDA_CHECK_LAUNCH();

  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
