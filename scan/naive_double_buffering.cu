#include <cmath>
#include <cuda_runtime.h>
#include <iostream>

// Error checking macro
#define CUDA_CHECK(ans)                                                        \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " at " << file
              << ":" << line << std::endl;
    if (abort)
      exit(code);
  }
}

// Naive inclusive scan using shared memory, 2 syncthreads per step
__global__ void naive_scan_shared_db(float *d_out, const float *d_in, int n) {
  __shared__ float buffer[2][1024]; // double buffering: ping-pong
  int tid = threadIdx.x;

  // Load input into buffer[0] (ping)
  if (tid < n)
    buffer[0][tid] = d_in[tid];
  __syncthreads();

  int curr = 0; // index of current read buffer

  for (int d = 1; d < n; d *= 2) {
    int prev = curr; // current read buffer
    curr = 1 - curr; // swap buffer: current write buffer

    if (tid < n) {
      float val = buffer[prev][tid];
      if (tid >= d)
        val += buffer[prev][tid - d];
      buffer[curr][tid] = val; // write to new buffer
    }

    __syncthreads(); // Ensure all threads finish before next iteration
  }

  // Write final result to global memory
  if (tid < n)
    d_out[tid] = buffer[curr][tid];
}

// CPU inclusive scan (reference)
void cpu_inclusive_scan(const float *input, float *output, int n) {
  output[0] = input[0];
  for (int i = 1; i < n; ++i)
    output[i] = output[i - 1] + input[i];
}

// Compare CPU and GPU output
bool check_equal(const float *a, const float *b, int n, float eps = 1e-5f) {
  for (int i = 0; i < n; ++i) {
    if (fabs(a[i] - b[i]) > eps) {
      std::cerr << "Mismatch at index " << i << ": " << a[i] << " (CPU) vs "
                << b[i] << " (GPU)\n";
      return false;
    }
  }
  return true;
}

int main() {
  cudaDeviceProp prop;
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  std::cout << "Running on GPU: " << prop.name << " (Compute Capability "
            << prop.major << "." << prop.minor << ")\n";
  CUDA_CHECK(cudaDeviceReset());

  const int n = 8;
  float h_in[n] = {1, 2, 3, 4, 5, 6, 7, 8};
  float h_out[n] = {0};
  float h_ref[n] = {0};

  float *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));

  // Launch with 1 block, n threads
  naive_scan_shared_db<<<1, n>>>(d_out, d_in, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(
      cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

  // CPU reference
  cpu_inclusive_scan(h_in, h_ref, n);

  std::cout << "Input:  ";
  for (int i = 0; i < n; i++)
    std::cout << h_in[i] << " ";
  std::cout << "\nGPU:    ";
  for (int i = 0; i < n; i++)
    std::cout << h_out[i] << " ";
  std::cout << "\nCPU Ref:";
  for (int i = 0; i < n; i++)
    std::cout << h_ref[i] << " ";
  std::cout << std::endl;

  if (check_equal(h_ref, h_out, n))
    std::cout << "✅ CPU and GPU results match.\n";
  else
    std::cout << "❌ CPU and GPU results differ.\n";

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
