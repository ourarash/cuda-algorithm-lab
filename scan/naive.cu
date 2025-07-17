#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

// Error checking macro
#define CUDA_CHECK(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " at " << file
              << ":" << line << std::endl;
    if (abort) exit(code);
  }
}

// Naive inclusive scan using shared memory, 2 syncthreads per step
__global__ void naive_scan_shared(float *d_out, const float *d_in, int n) {
  __shared__ float temp[1024];
  int tid = threadIdx.x;

  if (tid < n) {
    temp[tid] = d_in[tid];
  }
  __syncthreads();

  for (int d = 1; d < n; d *= 2) {
    float val = 0.0f;
    if (tid >= d && tid < n) {
      val = temp[tid - d];
    }

    __syncthreads();  // Ensure all reads before writes

    if (tid >= d && tid < n) {
      temp[tid] += val;
    }

    __syncthreads();  // Ensure all writes before next reads
  }

  if (tid < n) {
    d_out[tid] = temp[tid];
  }
}

// CPU inclusive scan (reference)
void cpu_inclusive_scan(const float *input, float *output, int n) {
  output[0] = input[0];
  for (int i = 1; i < n; ++i) output[i] = output[i - 1] + input[i];
}

// Compare CPU and GPU output
bool check_equal(const float *a, const float *b, int n, float eps = 1e-2f) {
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

  const int n = 1024;
  float *h_in = new float[n];
  float *h_out = new float[n];
  float *h_ref = new float[n];  // CPU reference output
  // Initialize input data with random numbers between 0 and 1.5
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(1.0f, 1.1f);

  for (int i = 0; i < n; i++) {
    h_in[i] = dis(gen);
  }

  float *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));

  // Launch with 1 block, n threads
  naive_scan_shared<<<1, n>>>(d_out, d_in, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(
      cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

  // CPU reference
  cpu_inclusive_scan(h_in, h_ref, n);

  printf("Last GPU output: %f\n", h_out[n - 1]);
  printf("Last CPU output: %f\n", h_ref[n - 1]);

  if (check_equal(h_ref, h_out, n)) {
    std::cout << "✅ CPU and GPU results match.\n";
  } else {
    std::cout << "❌ CPU and GPU results differ.\n";
  }

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
