#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#define cudaCheck(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    exit(code);
  }
}

__global__ void vector_reduce(float *in, size_t N) {
  int threadId = threadIdx.x;
  int index = 2 * blockIdx.x * blockDim.x + threadId;

  // Perform reduction within each block
  if (index + blockDim.x < N) {

    in[index] += in[index + blockDim.x];
  }
  __syncthreads(); // Ensure final result is written before global write

  if (threadId == 0) {
    // Store the result of the reduction in the first element of the block
    in[blockIdx.x] = in[blockIdx.x * blockDim.x];
  }
}

float cpu_reduce(float *in, size_t N) {
  float sum = std::accumulate(in, in + N, 0.0f);
  return sum;
}

int main() {
  int count;
  cudaError_t err = cudaGetDeviceCount(&count);
  printf("Devices: %d, Err: %s\n", count, cudaGetErrorString(err));

  int N = 1 << 20; // 1 million elements
  size_t size = N * sizeof(float);

  float *h_in = new float[N];
  float *d_in;
  cudaMalloc(&d_in, size);

  for (int i = 0; i < N; ++i) {
    h_in[i] = 1.0f; // Initialize input data
  }

  float sum = cpu_reduce(h_in, N);
  std::cout << "CPU reduction result: " << sum << std::endl;

  cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
  dim3 blockSize(256);

  size_t num = N;
  while (num > 1) {
    size_t blocks = (num + blockSize.x - 1) / blockSize.x;
    vector_reduce<<<blocks, blockSize>>>(d_in, num);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    num = blocks;
  }

  float *h_out = new float[1];
  cudaMemcpy(h_out, d_in, sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "Reduction result: " << *h_out << std::endl;
  cudaFree(d_in);
  delete[] h_in;
  delete[] h_out;
  return 0;
}