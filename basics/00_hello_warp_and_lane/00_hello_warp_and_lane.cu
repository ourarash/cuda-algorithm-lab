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
 *   - warp_id = threadIdx.x / 32
 *   - lane_id = threadIdx.x % 32
 * - Print those ids so the mapping from threads to warps is visible.
 */
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void hello_from_gpu() {
  int warp_id = threadIdx.x / 32; // Each warp contains 32 threads.
  int lane_id = threadIdx.x % 32; // Position within the warp.
  printf("Warp ID: %d, Lane ID: %d\n", warp_id, lane_id);
}

int main() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  int blockSize = 256; // define block size
  int numBlocksPerSM = 0;

  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM, hello_from_gpu,
                                                blockSize,
                                                0 // dynamic shared memory
  );

  printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
  printf("Max blocks per SM: %d\n", numBlocksPerSM);

  hello_from_gpu<<<1, 64>>>();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    return -1;
  }

  // Wait for GPU to finish before continuing
  cudaDeviceSynchronize();

  return 0;
}
