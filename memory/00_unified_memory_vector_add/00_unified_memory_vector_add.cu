/*
 * Unified Memory Vector Add
 *
 * Intention:
 * This example demonstrates CUDA managed memory by allocating vectors that are
 * directly accessible from both CPU and GPU code, with no explicit cudaMemcpy.
 *
 * High-Level Algorithm:
 * - Allocate two managed-memory arrays with cudaMallocManaged.
 * - Initialize them on the CPU.
 * - Launch a kernel that performs y[i] += x[i].
 * - Synchronize, then read and check the result directly on the CPU.
 *
 * The cudaDeviceSynchronize() before the CPU reads y is required: the kernel
 * launch is asynchronous, and the CPU must not touch managed memory the GPU
 * may still be writing.
 */
#include <cstdio>

#include "lab.cuh"

// CUDA kernel to add the elements of two arrays.
__global__ void add(int n, const float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] += x[i];
  }
}

int main(int argc, char **argv) {
  lab::Args args(argc, argv);
  const int n = static_cast<int>(args.get_int("n", args.quick() ? 4099 : 1 << 20));

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("Unified addressing: %s\n", prop.unifiedAddressing ? "Yes" : "No");
  printf("Managed memory:     %s\n", prop.managedMemory ? "Yes" : "No");
  printf("Concurrent managed access (GPU page faulting): %s\n",
         prop.concurrentManagedAccess ? "Yes" : "No");

  // Allocate unified memory, accessible from both CPU and GPU. Like every
  // runtime call, failure is reported through the return code.
  float *x = nullptr;
  float *y = nullptr;
  CUDA_CHECK(cudaMallocManaged(&x, n * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&y, n * sizeof(float)));

  // Initialize x and y on the host.
  for (int i = 0; i < n; ++i) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Launch the kernel using the managed pointers directly.
  const int block_size = 256;
  add<<<lab::ceil_div(n, block_size), block_size>>>(n, x, y);
  CUDA_CHECK_LAUNCH();

  // Wait for the GPU to finish before accessing the result on the CPU.
  CUDA_CHECK(cudaDeviceSynchronize());

  int wrong = 0;
  for (int i = 0; i < n; ++i) {
    if (y[i] != 3.0f) {
      ++wrong;
    }
  }
  printf("Checked %d elements, %d wrong\n", n, wrong);

  CUDA_CHECK(cudaFree(x));
  CUDA_CHECK(cudaFree(y));
  return lab::finish(wrong == 0);
}
