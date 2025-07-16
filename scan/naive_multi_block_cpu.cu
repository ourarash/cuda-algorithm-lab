#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

#define N 1024 * 1024         // Size of the input array
#define BLOCK_SIZE 1024  // Number of threads per block

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
__global__ void naive_scan_shared(double *d_out, double *partialSums,
                                  const double *d_in, int n) {
  __shared__ double temp[BLOCK_SIZE];
  int tid = threadIdx.x;
  int index = blockIdx.x * blockDim.x + tid;

  if (index < n) {
    temp[tid] = d_in[index];
  }
  __syncthreads();

  for (int d = 1; d < BLOCK_SIZE; d *= 2) {
    double val = 0.0f;
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
    d_out[index] = temp[tid];
  }
  if (tid == 0) {
    // Store the last value of the block for the next block's scan
    partialSums[blockIdx.x] = temp[BLOCK_SIZE - 1];
  }
}

// CPU inclusive scan (reference)
void cpu_inclusive_scan(const double *input, double *output, int n) {
  output[0] = input[0];
  for (int i = 1; i < n; ++i) {
    output[i] = output[i - 1] + input[i];
  }
}

// Compare CPU and GPU output
bool check_equal(const double *a, const double *b, int n, double eps = 1e-2f) {
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
  const int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  cudaDeviceProp prop;
  int device;

  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  std::cout << "Running on GPU: " << prop.name << " (Compute Capability "
            << prop.major << "." << prop.minor << ")\n";
  CUDA_CHECK(cudaDeviceReset());

  double *h_in = new double[N];
  double *h_out = new double[N];
  double *h_ref = new double[N];  // CPU reference output
  double *h_partialSums = new double[numBlocks];
  // Initialize input data with random numbers between 0 and 1.5
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(1.0f, 1.1f);

  for (int i = 0; i < N; i++) {
    h_in[i] = dis(gen);
  }

  double *d_in = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(double), cudaMemcpyHostToDevice));

  double *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(double)));

  double *d_partialSums = nullptr;
  CUDA_CHECK(cudaMalloc(&d_partialSums, numBlocks * sizeof(double)));

  // Launch the naive scan kernel
  dim3 block(BLOCK_SIZE);
  dim3 grid(numBlocks);
  naive_scan_shared<<<grid, block>>>(d_out, d_partialSums, d_in, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(
      cudaMemcpy(h_out, d_out, N * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_partialSums, d_partialSums, numBlocks * sizeof(double),
                        cudaMemcpyDeviceToHost));

  // Perform the final inclusive scan on the partial sums in the host.
  // TODO: do this on the GPU instead.
  cpu_inclusive_scan(h_partialSums, h_partialSums, numBlocks);

  // Add entry i to every element in block i+1.
  for (int i = BLOCK_SIZE; i < N; ++i) {
    int block_idx = i / BLOCK_SIZE;

    // Add the sum of all preceding blocks (0 to block_idx-1),
    // which is correctly stored in h_partialSums[block_idx - 1].
    h_out[i] += h_partialSums[block_idx - 1];
  }

  // CPU reference
  cpu_inclusive_scan(h_in, h_ref, N);

  printf("Last GPU output: %f\n", h_out[N - 1]);
  printf("Last CPU output: %f\n", h_ref[N - 1]);

  if (check_equal(h_ref, h_out, N)) {
    std::cout << "✅ CPU and GPU results match.\n";
  } else {
    std::cout << "❌ CPU and GPU results differ.\n";
  }

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
