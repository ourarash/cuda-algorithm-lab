/*
 * Reduction 12: CUB DeviceReduce (the baseline)
 *
 * Intention:
 * CUB is NVIDIA's library of tuned parallel primitives (it ships with the
 * CUDA Toolkit and underlies Thrust). cub::DeviceReduce::Sum is the number
 * the hand-written steps in this folder are measured against.
 *
 * Usage pattern: CUB device-wide algorithms take a temporary buffer whose
 * size depends on the problem. Call once with a null buffer to get the size,
 * then again with a buffer of at least that size to run. Production code
 * allocates the buffer once and reuses it; here it lives in the harness's
 * scratch space.
 */
#include <cub/cub.cuh>

#include "../reduction_harness.cuh"

void launch(float *d_in, int n, float *d_out, float *d_scratch) {
  size_t temp_bytes = 0;
  CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, temp_bytes, d_in, d_out, n));
  const size_t scratch_bytes = (static_cast<size_t>(n) + 1024) * sizeof(float);
  if (temp_bytes > scratch_bytes) {
    std::fprintf(stderr, "CUB needs %zu bytes of scratch, have %zu\n",
                 temp_bytes, scratch_bytes);
    std::exit(EXIT_FAILURE);
  }
  CUDA_CHECK(cub::DeviceReduce::Sum(d_scratch, temp_bytes, d_in, d_out, n));
}

int main(int argc, char **argv) {
  return run_reduction("12. CUB DeviceReduce::Sum", argc, argv, launch);
}
