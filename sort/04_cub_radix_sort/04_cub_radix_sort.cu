/*
 * CUB DeviceRadixSort (the baseline)
 *
 * Intention:
 * cub::DeviceRadixSort::SortKeys is NVIDIA's production radix sort (the
 * "Onesweep" algorithm on recent GPUs): 8-bit digits, tiles sorted in shared
 * memory before their coalesced global writes, and passes fused with a
 * decoupled look-back scan. It is the baseline for steps 02 and 03.
 */
#include <cub/cub.cuh>

#include "../sort_harness.cuh"

void launch(const unsigned int *d_in, unsigned int *d_out, int n,
            void *d_scratch, size_t scratch_bytes) {
  size_t temp_bytes = 0;
  CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes, d_in, d_out, n));
  if (temp_bytes > scratch_bytes) {
    std::fprintf(stderr, "CUB needs %zu bytes of scratch, have %zu\n",
                 temp_bytes, scratch_bytes);
    std::exit(EXIT_FAILURE);
  }
  CUDA_CHECK(cub::DeviceRadixSort::SortKeys(d_scratch, temp_bytes, d_in, d_out, n));
}

int main(int argc, char **argv) {
  return run_sort("CUB DeviceRadixSort::SortKeys", argc, argv, launch);
}
