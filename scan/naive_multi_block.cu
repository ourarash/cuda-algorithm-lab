#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

// Note that our solution assumes the second scan can be done in a single block!
// So adjust these numbers accordingly.
#define N 1024 * 1024     // Size of the input array
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
  // // Use blockDim.x instead of the BLOCK_SIZE macro for flexibility
  // extern __shared__ double temp[];
  // int tid = threadIdx.x;
  // int index = blockIdx.x * blockDim.x + tid;

  // // 1. SAFE LOAD: Load data if in bounds, otherwise load 0.0f
  // // This prevents uninitialized shared memory from corrupting the scan.
  // double val = (index < n) ? d_in[index] : 0.0f;
  // temp[tid] = val;
  // __syncthreads();

  // // The scan logic itself is mostly okay, but we can make it more robust.
  // // Using blockDim.x makes it independent of the compile-time macro.
  // for (int d = 1; d < blockDim.x; d *= 2) {
  //   // We can remove the bounds check (tid < n) inside the loop
  //   // because we padded with 0.0f, which won't affect the sum.
  //   val = (tid >= d) ? temp[tid - d] : 0.0f;
  //   __syncthreads();

  //   if (tid >= d) {
  //     temp[tid] += val;
  //   }
  //   __syncthreads();
  // }

  // // Write the output for valid threads
  // if (index < n) {
  //   d_out[index] = temp[tid];
  // }

  // // 2. CORRECT PARTIAL SUM: The last thread always holds the correct
  // // block sum because of the zero-padding.
  // if (tid == blockDim.x - 1) {
  //   partialSums[blockIdx.x] = temp[tid];
  // }

  extern __shared__ double temp[];
  int tid = threadIdx.x;
  int index = blockIdx.x * blockDim.x + tid;

  // 1. SAFE LOAD (same as before)
  temp[tid] = (index < n) ? d_in[index] : 0.0f;
  __syncthreads();

  // 2. UP-SWEEP (Reduction Phase)
  // Build a sum tree in shared memory
  for (int d = 1; d < blockDim.x; d *= 2) {
    if (tid % (2 * d) == (2 * d - 1)) {
      temp[tid] += temp[tid - d];
    }
    __syncthreads();
  }

  // 3. The last element now holds the total sum for the block
  if (tid == blockDim.x - 1) {
    // Save the sum for the next level scan
    if (partialSums) {
      partialSums[blockIdx.x] = temp[tid];
    }
    // Clear the last element for the down-sweep
    temp[tid] = 0.0f;
  }
  __syncthreads();

  // 4. DOWN-SWEEP (Distribution Phase)
  // Traverse down the tree, distributing partial sums
  for (int d = blockDim.x / 2; d > 0; d /= 2) {
    if (tid % (2 * d) == (2 * d - 1)) {
      double t = temp[tid - d];
      temp[tid - d] = temp[tid];
      temp[tid] += t;
    }
    __syncthreads();
  }

  // Write the final (inclusive) result
  if (index < n) {
    // The result is an exclusive scan; add the original value for inclusive
    d_out[index] = temp[tid] + ((index < n) ? d_in[index] : 0.0f);
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
bool check_equal(const double *a, const double *b, int n, double eps = 1e1f) {
  for (int i = 0; i < n; ++i) {
    if (fabs(a[i] - b[i]) >= eps) {
      std::cerr << "Mismatch at index " << i << ": " << a[i] << " (CPU) vs "
                << b[i] << " (GPU)\n";
      return false;
    }
  }
  return true;
}

int main() {
  const int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int numBlocksSecondRound = (numBlocks + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // Add this printf for debugging
  printf("INFO: N = %d, BLOCK_SIZE = %d\n", N, BLOCK_SIZE);
  printf("INFO: numBlocks (L1) = %d\n", numBlocks);
  printf("INFO: numBlocksSecondRound (L2) = %d\n\n", numBlocksSecondRound);

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
  size_t sharedMemSize = BLOCK_SIZE * sizeof(double);

  naive_scan_shared<<<grid, block, sharedMemSize>>>(d_out, d_partialSums, d_in,
                                                    N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(
      cudaMemcpy(h_out, d_out, N * sizeof(double), cudaMemcpyDeviceToHost));

  // Perform the final inclusive scan on the partial sums.
  double *d_partialSumsScan = nullptr;
  CUDA_CHECK(cudaMalloc(&d_partialSumsScan, numBlocks * sizeof(double)));

  double *d_partialSumsSecondRound = nullptr;
  CUDA_CHECK(cudaMalloc(&d_partialSumsSecondRound,
                        numBlocksSecondRound * sizeof(double)));
  double *h_partialSumsScan = new double[numBlocks];

  naive_scan_shared<<<numBlocksSecondRound, block, sharedMemSize>>>(
      d_partialSumsScan, d_partialSumsSecondRound, d_partialSums, numBlocks);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_partialSumsScan, d_partialSumsScan,
                        numBlocks * sizeof(double), cudaMemcpyDeviceToHost));

  // Get the sums of the "super-blocks" from the device
  double *h_partialSumsSecondRound = new double[numBlocksSecondRound];
  CUDA_CHECK(cudaMemcpy(h_partialSumsSecondRound, d_partialSumsSecondRound,
                        numBlocksSecondRound * sizeof(double),
                        cudaMemcpyDeviceToHost));

  // Add entry i to every element in block i+1.
  for (int i = BLOCK_SIZE; i < N; ++i) {
    int block_idx = i / BLOCK_SIZE;

    // Add the sum of all preceding blocks (0 to block_idx-1),
    // which is correctly stored in h_partialSums[block_idx - 1].
    h_out[i] += h_partialSumsScan[block_idx - 1];
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
  CUDA_CHECK(cudaFree(d_partialSums));
  CUDA_CHECK(cudaFree(d_partialSumsScan));
  CUDA_CHECK(cudaFree(d_partialSumsSecondRound));
  delete[] h_in;
  delete[] h_out;
  delete[] h_ref;
  delete[] h_partialSums;
  delete[] h_partialSumsScan;
  return 0;
}
