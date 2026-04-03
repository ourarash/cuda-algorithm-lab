/*
 * Hello Thread Hierarchy
 *
 * Intention:
 * This file is a minimal demonstration of CUDA's 3D thread-block indexing.
 *
 * High-Level Algorithm:
 * - Launch a 3D thread block.
 * - Each thread prints its local (x, y, z) coordinates.
 * - The output makes the thread hierarchy visible without any extra math.
 */
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void hello_from_gpu_2d() {
  printf("Hello from GPU thread (%d, %d, %d)\n", threadIdx.x, threadIdx.y,
         threadIdx.z);
}

int main() {
  dim3 threadsPerBlock(
      3, 4, 2);        // 24 threads per block spread across x, y, and z.
  dim3 numBlocks(2, 1); // 2 blocks in x, 1 in y

  hello_from_gpu_2d<<<numBlocks, threadsPerBlock>>>();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }

  cudaDeviceSynchronize();
  return 0;
}
