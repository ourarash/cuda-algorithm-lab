/*
 * CUB DeviceScan (the baseline)
 *
 * Intention:
 * cub::DeviceScan::InclusiveSum is NVIDIA's tuned implementation of the
 * single-pass decoupled look-back scan from step 08, with warp-parallel
 * look-back, tuned tile sizes per GPU, and vectorized loads. It is the number
 * steps 05-08 are measured against.
 *
 * As with every CUB device-wide algorithm, the first call (with a null
 * buffer) only reports how much temporary storage the scan needs.
 */
#include <cub/cub.cuh>

#include "../scan_harness.cuh"

void launch(const int *d_in, int *d_out, int n, void *d_scratch,
            size_t scratch_bytes) {
  size_t temp_bytes = 0;
  CUDA_CHECK(cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, d_in, d_out, n));
  if (temp_bytes > scratch_bytes) {
    std::fprintf(stderr, "CUB needs %zu bytes of scratch, have %zu\n",
                 temp_bytes, scratch_bytes);
    std::exit(EXIT_FAILURE);
  }
  CUDA_CHECK(cub::DeviceScan::InclusiveSum(d_scratch, temp_bytes, d_in, d_out, n));
}

int main(int argc, char **argv) {
  return run_scan("CUB DeviceScan::InclusiveSum", argc, argv, launch);
}
