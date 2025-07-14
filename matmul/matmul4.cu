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

// Kernel assumes:
// A: row-major FP16
// B: column-major FP16
// C: row-major FP32
__global__ void matMulKernelTiled(float *a, float *b, float *c, size_t N) {
  // Identify which tile of C this warp is computing
  int tileRow = blockIdx.y; // row tile index of C
  int tileCol = blockIdx.x; // col tile index of C

  // Declare fragments (register-level tiles)
  wmma::fragment<wmma::matrix_a, 16, 16, 8, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 8, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag;

  // Initialize output fragment to zero
  wmma::fill_fragment(c_frag, 0.0f);

  // Loop over K dimension in chunks of 8
  for (int k = 0; k < N; k += 8) {
    // Compute base indices in A and B for this tile
    int a_idx = tileRow * 16 * N + k; // row = tileRow * 16, col = k
    int b_idx = k * N + tileCol * 16; // row = k, col = tileCol * 16

    // Load 16×8 tile from A and 8×16 tile from B into fragments
    wmma::load_matrix_sync(a_frag, reinterpret_cast<half *>(a + a_idx), N);
    wmma::load_matrix_sync(b_frag, reinterpret_cast<half *>(b + b_idx), N);

    // Perform matrix multiply-accumulate: C += A × B
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  // Compute base index to write the 16×16 tile of C
  int c_idx = tileRow * 16 * N + tileCol * 16;

  // Store the output fragment to global memory
  wmma::store_matrix_sync(c + c_idx, c_frag, N, wmma::mem_row_major);
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