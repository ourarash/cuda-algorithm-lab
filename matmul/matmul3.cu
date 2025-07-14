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
__global__ void matMulKernelTiledVect4(float *a, float *b, float *c, size_t N) {
  __shared__ float tileA[TILE_SIZE][TILE_SIZE];
  __shared__ float4 tileB[TILE_SIZE][TILE_SIZE / 4];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x * 4;

  float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int tileIdx = 0; tileIdx < N / TILE_SIZE; ++tileIdx) {
    int aRow = row;
    int aCol = tileIdx * TILE_SIZE + threadIdx.x;
    int bRow = tileIdx * TILE_SIZE + threadIdx.y;
    int bCol = col;

    if (aRow < N && aCol < N)
      tileA[threadIdx.y][threadIdx.x] = a[aRow * N + aCol];
    else
      tileA[threadIdx.y][threadIdx.x] = 0.0f;

    if (bRow < N && bCol + 3 < N)
      tileB[threadIdx.y][threadIdx.x] = reinterpret_cast<float4 *>(
          &b[bRow * N])[threadIdx.x + col / 4 - tileIdx * TILE_SIZE / 4];
    else
      tileB[threadIdx.y][threadIdx.x] = make_float4(0, 0, 0, 0);

    __syncthreads();

    for (int k = 0; k < TILE_SIZE; ++k) {
      float aVal = tileA[threadIdx.y][k];
      float4 bVal = tileB[k][threadIdx.x];
      acc.x += aVal * bVal.x;
      acc.y += aVal * bVal.y;
      acc.z += aVal * bVal.z;
      acc.w += aVal * bVal.w;
    }

    __syncthreads();
  }

  if (row < N && col + 3 < N)
    reinterpret_cast<float4 *>(&c[row * N])[col / 4] = acc;
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
  matMulKernelTiledVect4<<<gridSize, blockSize>>>(d_a, d_b, d_c, MATRIX_SIZE);

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