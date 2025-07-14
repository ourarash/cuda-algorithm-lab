#include <cuda_runtime.h>
#include <iostream>

int main() {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  std::cout << "Device count: " << count << "\n";
  std::cout << "CUDA error: " << cudaGetErrorString(err) << "\n";
  return 0;
}
