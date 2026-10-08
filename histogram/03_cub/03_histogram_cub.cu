/*
 * Histogram 3: CUB DeviceHistogram (the baseline)
 *
 * Intention:
 * cub::DeviceHistogram::HistogramEven bins samples into equal-width bins.
 * With 257 bin boundaries 0, 1, ..., 256, every byte value gets its own bin.
 * CUB picks privatization, aggregation, and tile sizes per GPU.
 */
#include <cub/cub.cuh>

#include "../histogram_harness.cuh"

void launch(const unsigned char *d_in, int n, unsigned int *d_hist,
            void *d_scratch, size_t scratch_bytes) {
  const int num_levels = NUM_BINS + 1;
  const int lower = 0;
  const int upper = NUM_BINS;
  size_t temp_bytes = 0;
  CUDA_CHECK(cub::DeviceHistogram::HistogramEven(
      nullptr, temp_bytes, d_in, d_hist, num_levels, lower, upper, n));
  if (temp_bytes > scratch_bytes) {
    std::fprintf(stderr, "CUB needs %zu bytes of scratch, have %zu\n",
                 temp_bytes, scratch_bytes);
    std::exit(EXIT_FAILURE);
  }
  CUDA_CHECK(cub::DeviceHistogram::HistogramEven(
      d_scratch, temp_bytes, d_in, d_hist, num_levels, lower, upper, n));
}

int main(int argc, char **argv) {
  return run_histogram("3. CUB DeviceHistogram", argc, argv, launch);
}
