#include <stdio.h>

__global__ void hello_from_gpu_2d() {
  printf("Hello from GPU thread (%d, %d, %d)\n", threadIdx.x, threadIdx.y,
         threadIdx.z);
}

int main() {
  dim3 threadsPerBlock(
      300, 4, 2);       // 3 threads in x, 4 in y (total 12 threads per block)
  dim3 numBlocks(2, 1); // 2 blocks in x, 1 in y

  hello_from_gpu_2d<<<numBlocks, threadsPerBlock>>>();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }

  cudaDeviceSynchronize();
  return 0;
}
