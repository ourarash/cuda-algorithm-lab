#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
// CUDA kernel to add elements of two arrays
__global__ void add(int n, float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] += x[i];
}

int main(void) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("Unified addressing: %s\n", prop.unifiedAddressing ? "Yes" : "No");
  printf("Managed memory:     %s\n", prop.managedMemory ? "Yes" : "No");

  int N = 1 << 3;
  float *x, *y;
  // Allocate Unified Memory -- accessible from CPU or GPU
  cudaMallocManaged(&x, N * sizeof(float));
  if (!x) {
    printf("Alloc failed 1\n");
    return -1;
  }

  cudaMallocManaged(&y, N * sizeof(float));
  if (!x) {
    printf("Alloc failed 2\n");
    return -1;
  }

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
  // Launch kernel on 1M elements on the GPU
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  add<<<numBlocks, blockSize>>>(N, x, y);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  // Free memory
  cudaFree(x);
  cudaFree(y);
  return 0;
}
