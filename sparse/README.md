# Sparse Matrices

| Step | Format / method | Matrix |
| --- | --- | --- |
| [00_spmv_coo](00_spmv_coo/00_spmv_coo.cu) | COO, one thread per nonzero, atomics | Small, uniform random |
| [01_spmv_csr](01_spmv_csr/01_spmv_csr.cu) | CSR, one thread per row | Small, uniform random |
| [02_spmv_ell](02_spmv_ell/02_spmv_ell.cu) | ELL, column-major padding | Small, uniform random |
| [03_spgemm_cusparse](03_spgemm_cusparse/03_spgemm_cusparse.cu) | Sparse x sparse with cuSPARSE | Small, uniform random |
| [04_spmv_csr_scalar_power_law](04_spmv_csr_scalar_power_law/04_spmv_csr_scalar_power_law.cu) | CSR scalar | Large, power-law rows |
| [05_spmv_csr_vector](05_spmv_csr_vector/05_spmv_csr_vector.cu) | CSR vector, one warp per row | Large, power-law rows |
| [06_spmv_hybrid](06_spmv_hybrid/06_spmv_hybrid.cu) | Hybrid ELL + COO | Large, power-law rows |
| [07_spmv_cusparse](07_spmv_cusparse/07_spmv_cusparse.cu) | cuSPARSE SpMV, the baseline | Large, power-law rows |

Steps 04-07 share [spmv_harness.cuh](spmv_harness.cuh). Its generated matrix
has power-law row lengths (most rows short, a few with thousands of
nonzeros), which is what exposes load imbalance. Run any of them on a real
matrix in Matrix Market format with `--mtx path/to/matrix.mtx`, for example
from the [SuiteSparse Matrix Collection](https://sparse.tamu.edu).
