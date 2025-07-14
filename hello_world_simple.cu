#include <stdio.h>

__global__ void hello_from_gpu() {
  int warp_id = threadIdx.x / 32; // Each warp has 32 threads
  int lane_id = threadIdx.x % 32; // Lane ID within the warp
  // printf("Hello from GPU thread (%d, %d)\n", blockIdx.x, threadIdx.x);
  printf("Warp ID: %d, Lane ID: %d\n", warp_id, lane_id);
  // printf("Block ID: %d, Thread ID: %d\n", blockIdx.x, threadIdx.x);
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

  hello_from_gpu<<<2, 2>>>();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    return -1;
  }

  // Wait for GPU to finish before continuing
  cudaDeviceSynchronize();

  return 0;
}
