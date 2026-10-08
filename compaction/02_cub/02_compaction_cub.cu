/*
 * Stream Compaction 2: CUB DeviceSelect::If (the baseline)
 *
 * Intention:
 * cub::DeviceSelect::If is a stable, single-pass compaction (it uses the
 * decoupled look-back scan from scan/08 internally). It is the baseline for
 * steps 0 and 1.
 */
#include <cub/cub.cuh>

#include "../compaction_harness.cuh"

struct KeepOp {
  __host__ __device__ bool operator()(const int &x) const { return keep(x); }
};

void launch(const int *d_in, int *d_out, int *d_count, int n, void *d_scratch,
            size_t scratch_bytes) {
  size_t temp_bytes = 0;
  CUDA_CHECK(cub::DeviceSelect::If(nullptr, temp_bytes, d_in, d_out, d_count, n,
                                   KeepOp()));
  if (temp_bytes > scratch_bytes) {
    std::fprintf(stderr, "CUB needs %zu bytes of scratch, have %zu\n",
                 temp_bytes, scratch_bytes);
    std::exit(EXIT_FAILURE);
  }
  CUDA_CHECK(cub::DeviceSelect::If(d_scratch, temp_bytes, d_in, d_out, d_count,
                                   n, KeepOp()));
}

int main(int argc, char **argv) {
  return run_compaction("2. CUB DeviceSelect::If", argc, argv, launch,
                        /*stable=*/true);
}
