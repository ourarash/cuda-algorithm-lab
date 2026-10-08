/*
 * SpMV 6: Hybrid ELL + COO
 *
 * Intention:
 * ELL (step 02) is fast when rows have similar lengths: column-major padding
 * gives coalesced loads and every thread does the same amount of work. But
 * padding every row to the longest row of a power-law matrix would be
 * enormous. The hybrid format (Bell and Garland, 2008) stores the first K
 * entries of every row in ELL and the overflow in COO:
 * - K is chosen so that 90% of rows fit entirely in the ELL part.
 * - ELL kernel: one thread per row over its (up to) K slots, writes y[row].
 * - COO kernel: one thread per overflow entry, atomicAdd(&y[row], v * x[c]).
 *   The few long rows are spread across many threads instead of one.
 *
 * The CSR-to-hybrid conversion runs on the host once in setup().
 */
#include "../spmv_harness.cuh"

constexpr int THREADS = 256;

static int g_rows = 0;
static int g_width = 0;     // K, ELL slots per row
static int g_coo_nnz = 0;
static int *d_ell_col = nullptr;
static float *d_ell_val = nullptr;
static int *d_coo_row = nullptr;
static int *d_coo_col = nullptr;
static float *d_coo_val = nullptr;

__global__ void spmv_ell(int rows, int width, const int *col, const float *val,
                         const float *x, float *y) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows) {
    float sum = 0.0f;
    for (int slot = 0; slot < width; ++slot) {
      const int idx = slot * rows + row;  // Column-major: coalesced
      sum += val[idx] * x[col[idx]];      // Padding has value 0
    }
    y[row] = sum;
  }
}

__global__ void spmv_coo_add(int nnz, const int *row, const int *col,
                             const float *val, const float *x, float *y) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < nnz) {
    atomicAdd(&y[row[k]], val[k] * x[col[k]]);
  }
}

void setup(const CsrHost &A, const CsrDevice &) {
  g_rows = A.rows;
  std::vector<int> lengths(A.rows);
  for (int r = 0; r < A.rows; ++r) lengths[r] = A.row_ptr[r + 1] - A.row_ptr[r];
  std::vector<int> sorted = lengths;
  std::sort(sorted.begin(), sorted.end());
  g_width = sorted.empty() ? 0 : sorted[static_cast<size_t>(0.9 * (sorted.size() - 1))];

  std::vector<int> ell_col(static_cast<size_t>(g_width) * A.rows, 0);
  std::vector<float> ell_val(static_cast<size_t>(g_width) * A.rows, 0.0f);
  std::vector<int> coo_row, coo_col;
  std::vector<float> coo_val;
  for (int r = 0; r < A.rows; ++r) {
    for (int k = A.row_ptr[r], slot = 0; k < A.row_ptr[r + 1]; ++k, ++slot) {
      if (slot < g_width) {
        ell_col[static_cast<size_t>(slot) * A.rows + r] = A.col_idx[k];
        ell_val[static_cast<size_t>(slot) * A.rows + r] = A.vals[k];
      } else {
        coo_row.push_back(r);
        coo_col.push_back(A.col_idx[k]);
        coo_val.push_back(A.vals[k]);
      }
    }
  }
  g_coo_nnz = static_cast<int>(coo_val.size());
  std::printf("Hybrid: ELL width K = %d (%zu slots, %.0f%% useful), %d entries in COO\n",
              g_width, ell_val.size(),
              ell_val.empty() ? 0.0 : 100.0 * (A.nnz() - g_coo_nnz) / ell_val.size(),
              g_coo_nnz);

  const size_t ell = std::max<size_t>(1, ell_val.size());
  const size_t coo = std::max<size_t>(1, coo_val.size());
  CUDA_CHECK(cudaMalloc(&d_ell_col, ell * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_ell_val, ell * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_coo_row, coo * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_coo_col, coo * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_coo_val, coo * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_ell_col, ell_col.data(), ell_col.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_ell_val, ell_val.data(), ell_val.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_coo_row, coo_row.data(), coo_row.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_coo_col, coo_col.data(), coo_col.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_coo_val, coo_val.data(), coo_val.size() * sizeof(float), cudaMemcpyHostToDevice));
}

void run(const float *x, float *y) {
  spmv_ell<<<lab::ceil_div(g_rows, THREADS), THREADS>>>(g_rows, g_width, d_ell_col,
                                                        d_ell_val, x, y);
  if (g_coo_nnz > 0) {
    spmv_coo_add<<<lab::ceil_div(g_coo_nnz, THREADS), THREADS>>>(
        g_coo_nnz, d_coo_row, d_coo_col, d_coo_val, x, y);
  }
}

int main(int argc, char **argv) {
  const int result = run_spmv("6. Hybrid ELL + COO", argc, argv, setup, run);
  cudaFree(d_ell_col);
  cudaFree(d_ell_val);
  cudaFree(d_coo_row);
  cudaFree(d_coo_col);
  cudaFree(d_coo_val);
  return result;
}
