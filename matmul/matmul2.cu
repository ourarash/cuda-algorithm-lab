#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

#define MATRIX_SIZE                                                            \
  1024 // Size of the matrix (N x N) - 1024x1024 is more reasonable
#define TILE_SIZE 32 // Tile size for shared memory

/**
 * Matrix multiplication on CPU.
 */
void cpuMatMul(float *a, float *b, float *c, size_t N = MATRIX_SIZE) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      c[i * N + j] = 0.0f;
      for (int k = 0; k < N; ++k) {
        c[i * N + j] += a[i * N + k] * b[k * N + j];
      }
    }
  }
}

__global__ void matMulKernel(float *a, float *b, float *c,
                             size_t N = MATRIX_SIZE) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < N && col < N) {
    float value = 0.0f;
    for (int k = 0; k < N; ++k) {
      value += a[row * N + k] * b[k * N + col];
    }
    c[row * N + col] = value;
  }
}

__global__ void matMulKernelTiled(float *a, float *b, float *c, size_t N) {
  __shared__ float tileA[TILE_SIZE][TILE_SIZE];
  __shared__ float tileB[TILE_SIZE][TILE_SIZE+1];

  // Global row and column index in the output matrix C
  int globalRow = blockIdx.y * TILE_SIZE + threadIdx.y;
  int globalCol = blockIdx.x * TILE_SIZE + threadIdx.x;

  float partialSum = 0.0f;

  // Loop over tiles along the shared dimension
  for (int tileIdx = 0; tileIdx < N / TILE_SIZE; tileIdx++) {
    // Compute global indices for the current tile of A
    int aRow = globalRow;
    int aCol = tileIdx * TILE_SIZE + threadIdx.x;

    // Compute global indices for the current tile of B
    int bRow = tileIdx * TILE_SIZE + threadIdx.y;
    int bCol = globalCol;

    // Load elements into shared memory (with bounds check if needed)
    tileA[threadIdx.y][threadIdx.x] = a[aRow * N + aCol];
    tileB[threadIdx.y][threadIdx.x] = b[bRow * N + bCol];

    __syncthreads(); // Wait for all threads to load their data

    // Multiply row of A with column of B
    for (int k = 0; k < TILE_SIZE; k++) {
      partialSum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
    }

    __syncthreads(); // Wait before loading next tile
  }

  // Write the result to global memory
  c[globalRow * N + globalCol] = partialSum;
}

/**
 * Compare two arrays of floats with tolerance for floating point precision
 * errors. Returns true if arrays match within the specified tolerance.
 */
bool compareArraysWithTolerance(float *gpu_result, float *cpu_result,
                                size_t size, float epsilon = 1e-5f,
                                int max_errors_to_show = 10) {
  bool results_match = true;
  int mismatches = 0;

  for (int i = 0; i < size; ++i) {
    float diff = fabs(gpu_result[i] - cpu_result[i]);
    float relative_error =
        (cpu_result[i] != 0.0f) ? diff / fabs(cpu_result[i]) : diff;

    if (diff > epsilon && relative_error > epsilon) {
      results_match = false;
      if (mismatches < max_errors_to_show) {
        std::cerr << "Mismatch at index " << i << ": " << gpu_result[i]
                  << " (GPU) vs " << cpu_result[i] << " (CPU), diff = " << diff
                  << ", relative error = " << relative_error << std::endl;
      }
      mismatches++;
    }
  }

  if (results_match) {
    std::cout << "Results match within tolerance (epsilon = " << epsilon << ")"
              << std::endl;
  } else {
    std::cerr << "Found " << mismatches << " mismatches out of " << size
              << " elements" << std::endl;
  }

  return results_match;
}

void printSharedMemoryInfo() {
  int device;
  cudaGetDevice(&device);

  int sharedPerSM;
  cudaDeviceGetAttribute(&sharedPerSM,
                         cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);

  printf("Shared memory per SM: %d bytes\n", sharedPerSM);
}
int main() {
  // printSharedMemoryInfo();
  float *h_a = new float[(size_t)MATRIX_SIZE * MATRIX_SIZE];
  float *h_b = new float[(size_t)MATRIX_SIZE * MATRIX_SIZE];
  float *h_c = new float[(size_t)MATRIX_SIZE * MATRIX_SIZE];

  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, (size_t)MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
  cudaMalloc(&d_b, (size_t)MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
  cudaMalloc(&d_c, (size_t)MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

  // Initialize matrices on the host
  for (size_t i = 0; i < (size_t)MATRIX_SIZE * MATRIX_SIZE; ++i) {
    h_a[i] = static_cast<float>(i);
    h_b[i] = static_cast<float>(i);
    h_c[i] = 0.0f;
  }

  // Copy matrices from host to device
  cudaMemcpy(d_a, h_a, (size_t)MATRIX_SIZE * MATRIX_SIZE * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, (size_t)MATRIX_SIZE * MATRIX_SIZE * sizeof(float),
             cudaMemcpyHostToDevice);

  // Launch kernel
  dim3 blockSize(TILE_SIZE, TILE_SIZE);
  dim3 gridSize((MATRIX_SIZE + blockSize.x - 1) / blockSize.x,
                (MATRIX_SIZE + blockSize.y - 1) / blockSize.y);
  matMulKernelTiled<<<gridSize, blockSize>>>(d_a, d_b, d_c, MATRIX_SIZE);

  // Copy result from device to host
  cudaMemcpy(h_c, d_c, (size_t)MATRIX_SIZE * MATRIX_SIZE * sizeof(float),
             cudaMemcpyDeviceToHost);

  // Verify the result on the CPU
  float *h_c_cpu = new float[(size_t)MATRIX_SIZE * MATRIX_SIZE];
  cpuMatMul(h_a, h_b, h_c_cpu);

  // Compare results with floating point tolerance
  compareArraysWithTolerance(h_c, h_c_cpu, (size_t)MATRIX_SIZE * MATRIX_SIZE);

  std::cout << "Matrix multiplication completed successfully!" << std::endl;

  // Clean up
  delete[] h_a;
  delete[] h_b;
  delete[] h_c;
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}