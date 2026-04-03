/*
 * Runtime API Device Query
 *
 * Intention:
 * This file is a compact device-query example that prints the most useful
 * CUDA runtime properties for every visible GPU.
 *
 * High-Level Algorithm:
 * - Ask the runtime how many CUDA devices are present.
 * - Loop over the devices.
 * - Query and print basic architectural properties such as SM count, warp
 *   size, memory sizes, and compute capability.
 */
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
  printf("Hello from the CUDA runtime API!\n");

  // Check if CUDA is available
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    printf("No CUDA devices found.\n");
    return 1;
  }

  printf("Number of CUDA devices: %d\n", deviceCount);

  // Print device properties for each device
  for (int i = 0; i < deviceCount; ++i) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device %d: %s\n", i, prop.name);
    printf("  Total global memory: %zu bytes\n", prop.totalGlobalMem);
    printf("  Multiprocessor count: %d\n", prop.multiProcessorCount);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max threads per multiprocessor: %d\n",
           prop.maxThreadsPerMultiProcessor);
    printf("  Max grid size: (%d, %d, %d)\n", prop.maxGridSize[0],
           prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Clock rate: %d kHz\n", prop.clockRate);
    printf("  Memory clock rate: %d kHz\n", prop.memoryClockRate);
    printf("  Memory bus width: %d bits\n", prop.memoryBusWidth);
    printf("  L2 cache size: %zu bytes\n", prop.l2CacheSize);
    printf("  Total constant memory: %zu bytes\n", prop.totalConstMem);
    printf("  Max texture dimensions: %d x %d x %d\n", prop.maxTexture1D,
           prop.maxTexture2D[0], prop.maxTexture2D[1]);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max shared memory per block: %zu bytes\n",
           prop.sharedMemPerBlock);
    printf("  Warp size: %d\n", prop.warpSize);
  }

  return 0;
}
