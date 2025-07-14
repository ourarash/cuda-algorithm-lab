#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

using namespace std;

__global__ void vector_add(const float *a, const float *b, float *c, int N) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
    c[i] = a[i] + b[i];
}

int main() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
  printf("Max warps per SM: %d\n", prop.maxThreadsPerMultiProcessor / 32);

  const int N = 1 << 20; // 1 million elements

  size_t size = N * sizeof(float);

  float *h_a = (float *)malloc(size);
  float *h_b = (float *)malloc(size);
  float *h_c = (float *)malloc(size);

  for (int i = 0; i < N; ++i) {
    h_a[i] = 1.0f;
    h_b[i] = 2.0f;
  }

  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);

  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

  dim3 blockSize(1024);
  // Ceiling division to calculate the number of blocks needed
  // ceil(a/b) = (a + b - 1) / b (for positive integers)
  dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

  std::cout << "Grid size: " << gridSize.x << ", Block size: " << blockSize.x
            << std::endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  vector_add<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
  cudaEventRecord(stop);
  cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

  cudaError_t err = cudaGetLastError();
if (err != cudaSuccess)
    std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << "\n";
cudaDeviceSynchronize(); // ensures kernel finishes

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Time taken: %f ms\n", milliseconds);

  printf("Done. Sample output: c[N-1] = %f\n", h_c[N - 1]);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  free(h_a);
  free(h_b);
  free(h_c);

  return 0;
}
