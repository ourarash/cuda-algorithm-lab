/*
 * SpMV 5: CSR Vector (One Warp per Row)
 *
 * Intention:
 * Assign a whole warp to each row instead of one thread (Bell and Garland,
 * "Efficient Sparse Matrix-Vector Multiplication on CUDA", 2008):
 * - The 32 lanes walk the row's nonzeros together, lane k taking entries
 *   k, k + 32, ..., so the loads of col_idx and vals are coalesced.
 * - A long row is split across 32 lanes, so it takes 1/32 of the time.
 * - The lanes' partial sums are combined with warp shuffles; lane 0 writes
 *   y[row].
 *
 * Trade-off: rows shorter than 32 leave lanes idle. With an average of about
 * 12 nonzeros per row most lanes do nothing, so this wins on long rows and
 * loses on short ones. Production kernels choose the number of lanes per
 * row from the average row length (CSR-adaptive, merge-based SpMV); step 06
 * handles long rows separately instead.
 */
#include "../spmv_harness.cuh"

constexpr int THREADS = 256;
constexpr unsigned FULL_MASK = 0xFFFFFFFFu;
static CsrDevice A;

__global__ void spmv_csr_vector(int rows, const int *row_ptr, const int *col_idx,
                                const float *vals, const float *x, float *y) {
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x % 32;
  if (row >= rows) {
    return;  // Uniform per warp: all 32 lanes share the row
  }
  float sum = 0.0f;
  for (int k = row_ptr[row] + lane; k < row_ptr[row + 1]; k += 32) {
    sum += vals[k] * x[col_idx[k]];
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(FULL_MASK, sum, offset);
  }
  if (lane == 0) {
    y[row] = sum;
  }
}

void setup(const CsrHost &, const CsrDevice &device) { A = device; }

void run(const float *x, float *y) {
  const long long threads = 32LL * A.rows;
  const int blocks = static_cast<int>((threads + THREADS - 1) / THREADS);
  spmv_csr_vector<<<blocks, THREADS>>>(A.rows, A.row_ptr, A.col_idx, A.vals, x, y);
}

int main(int argc, char **argv) {
  return run_spmv("5. CSR vector (warp per row)", argc, argv, setup, run);
}
