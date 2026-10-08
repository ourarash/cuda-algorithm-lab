/*
 * SpMV 4: CSR Scalar on an Irregular Matrix
 *
 * Intention:
 * The CSR scalar kernel from step 01 (one thread per row), run on a large
 * matrix with power-law row lengths (see ../spmv_harness.cuh). It is the
 * baseline for steps 05-07.
 *
 * What goes wrong on irregular matrices:
 * - Load imbalance: a warp finishes only when its longest row does. One row
 *   with thousands of nonzeros keeps 31 idle threads waiting.
 * - Uncoalesced loads: neighbouring threads read different rows, so their
 *   loads of col_idx and vals are far apart in memory.
 */
#include "../spmv_harness.cuh"

constexpr int THREADS = 256;
static CsrDevice A;

__global__ void spmv_csr_scalar(int rows, const int *row_ptr, const int *col_idx,
                                const float *vals, const float *x, float *y) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows) {
    float sum = 0.0f;
    for (int k = row_ptr[row]; k < row_ptr[row + 1]; ++k) {
      sum += vals[k] * x[col_idx[k]];
    }
    y[row] = sum;
  }
}

void setup(const CsrHost &, const CsrDevice &device) { A = device; }

void run(const float *x, float *y) {
  spmv_csr_scalar<<<lab::ceil_div(A.rows, THREADS), THREADS>>>(
      A.rows, A.row_ptr, A.col_idx, A.vals, x, y);
}

int main(int argc, char **argv) {
  return run_spmv("4. CSR scalar (thread per row)", argc, argv, setup, run);
}
