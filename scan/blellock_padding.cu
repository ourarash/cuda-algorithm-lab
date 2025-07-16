#include <cmath>
#include <cuda_runtime.h>
#include <iostream>

#define NUM_BANKS 32
// i is changed to i + (i / NUM_BANKS)
#define CONFLICT_FREE_OFFSET(i) ((i) + ((i) / NUM_BANKS))

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

__global__ void blelloch_scan_shared(float *d_out, const float *d_in, int n) {
  __shared__ float temp[1024]; // shared buffer for 1 block
  int tid = threadIdx.x;

  // Load input into shared memory
  if (tid < n) {
    temp[tid] = d_in[tid];
  } else {
    temp[tid] = 0.0f; // fill unused threads with 0
  }
  __syncthreads();

  int offset;
  // Up-Sweep (reduce) phase
  for (offset = 1; offset < n; offset *= 2) {
    // maps thread tid to position i at level d in a virtual binary tree.
    // Each thread computes the sum of its own value and the value at
    // position i - offset, if it exists.
    // This maps thread IDs to the correct node index in a conceptual binary
    // tree built over an array, where each level of the tree corresponds to a
    // log₂(n) scan step.
    int i = (tid + 1) * offset * 2 - 1;
    if (i < n)
      temp[i] += temp[i - offset];
    __syncthreads();
  }

  // Set root to 0 for exclusive scan
  if (tid == 0)
    temp[n - 1] = 0.0f;
  __syncthreads();

  // Down-Sweep phase
  for (offset = n / 2; offset >= 1; offset /= 2) {
    // We want thread 0 to be mapped to the root of the tree, then thread 1 to
    // the root of the other tree at the same level.
    // At Iteration 1, There is only one tree (offset = n/2), so i = 1(n/2)*2-1
    // = n-1. If n=8, i = 7.
    //
    // At Iteration 2, There are two trees (offset = n/4), so i = 1(n/4)*2-1 =
    // n/2-1 and i = 2(n/4)*2-1 = n-1. If n=8, i = 3 and i = 7.
    int bi = (tid + 1) * offset * 2 - 1; // parent (right) index
    int ai = bi - offset;                // left child index

    if (bi < n) {
      float t = temp[CONFLICT_FREE_OFFSET(ai)]; // save old left child
      temp[CONFLICT_FREE_OFFSET(ai)] =
          temp[CONFLICT_FREE_OFFSET(bi)];  // move parent value to left
      temp[CONFLICT_FREE_OFFSET(bi)] += t; // update right = parent + old left
    }

    __syncthreads();
  }

  // Write result to global memory
  if (tid < n) {
    d_out[tid] = temp[tid];
  }
}

// CPU inclusive scan (reference)
void cpu_inclusive_scan(const float *input, float *output, int n) {
  output[0] = input[0];
  for (int i = 1; i < n; ++i)
    output[i] = output[i - 1] + input[i];
}

void cpu_exclusive_scan(const float *input, float *output, int n) {
  output[0] = 0.0f; // exclusive scan starts with 0
  for (int i = 1; i < n; ++i)
    output[i] = output[i - 1] + input[i - 1];
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
  blelloch_scan_shared<<<1, n>>>(d_out, d_in, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(
      cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

  // CPU reference
  cpu_exclusive_scan(h_in, h_ref, n);

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
