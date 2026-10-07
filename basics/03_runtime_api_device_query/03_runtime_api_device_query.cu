/*
 * Runtime API Device Query
 *
 * Intention:
 * This file is a compact device-query example that prints the CUDA runtime
 * properties that matter most when writing and tuning kernels, for every
 * visible GPU.
 *
 * High-Level Algorithm:
 * - Ask the runtime how many CUDA devices are present.
 * - Loop over the devices.
 * - Read the cudaDeviceProp struct for the common properties, and use
 *   cudaDeviceGetAttribute for the rest.
 * - Derive the theoretical peak DRAM bandwidth from the memory clock and bus
 *   width; bandwidth-bound kernels are measured against this number.
 *
 * Note: CUDA 13 removed several deprecated cudaDeviceProp fields, including
 * clockRate and memoryClockRate. cudaDeviceGetAttribute works on every
 * toolkit version, so it is used for those values here.
 */
#include <cstdio>

#include "lab.cuh"

static int attribute(cudaDeviceAttr attr, int device) {
  int value = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&value, attr, device));
  return value;
}

int main() {
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    printf("No CUDA devices found.\n");
    return 1;
  }

  int runtime_version = 0;
  int driver_version = 0;
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
  CUDA_CHECK(cudaDriverGetVersion(&driver_version));
  printf("CUDA runtime %d.%d, driver supports up to %d.%d\n",
         runtime_version / 1000, (runtime_version % 1000) / 10,
         driver_version / 1000, (driver_version % 1000) / 10);
  printf("Number of CUDA devices: %d\n", device_count);

  for (int i = 0; i < device_count; ++i) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
    const int sm_clock_khz = attribute(cudaDevAttrClockRate, i);
    const int mem_clock_khz = attribute(cudaDevAttrMemoryClockRate, i);
    const double peak_gbs =
        2.0 * mem_clock_khz * 1e3 * (prop.memoryBusWidth / 8.0) / 1e9;

    printf("\nDevice %d: %s\n", i, prop.name);
    printf("  Compute capability:            %d.%d\n", prop.major, prop.minor);
    printf("  Multiprocessors (SMs):         %d\n", prop.multiProcessorCount);
    printf("  SM clock:                      %.0f MHz\n", sm_clock_khz / 1e3);
    printf("  Warp size:                     %d\n", prop.warpSize);

    printf("  Max threads per block:         %d\n", prop.maxThreadsPerBlock);
    printf("  Max threads per SM:            %d (%d warps)\n",
           prop.maxThreadsPerMultiProcessor,
           prop.maxThreadsPerMultiProcessor / prop.warpSize);
    printf("  Max blocks per SM:             %d\n",
           prop.maxBlocksPerMultiProcessor);
    printf("  Max block dimensions:          (%d, %d, %d)\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  Max grid dimensions:           (%d, %d, %d)\n", prop.maxGridSize[0],
           prop.maxGridSize[1], prop.maxGridSize[2]);

    printf("  32-bit registers per SM:       %d\n", prop.regsPerMultiprocessor);
    printf("  32-bit registers per block:    %d\n", prop.regsPerBlock);
    printf("  Shared memory per SM:          %zu KB\n",
           prop.sharedMemPerMultiprocessor / 1024);
    printf("  Shared memory per block:       %zu KB (opt-in max %zu KB)\n",
           prop.sharedMemPerBlock / 1024, prop.sharedMemPerBlockOptin / 1024);
    printf("  L2 cache:                      %d KB\n", prop.l2CacheSize / 1024);
    printf("  Constant memory:               %zu KB\n", prop.totalConstMem / 1024);

    printf("  Global memory:                 %.1f GB\n",
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Memory clock:                  %.0f MHz\n", mem_clock_khz / 1e3);
    printf("  Memory bus width:              %d bits\n", prop.memoryBusWidth);
    printf("  Peak DRAM bandwidth:           %.0f GB/s\n", peak_gbs);
    printf("  Max 1D / 2D texture size:      %d / %d x %d\n", prop.maxTexture1D,
           prop.maxTexture2D[0], prop.maxTexture2D[1]);
  }

  return 0;
}
