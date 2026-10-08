/*
 * Debugging 1: Shared-Memory Race (compute-sanitizer racecheck)
 *
 * The bug: a tree reduction in shared memory without __syncthreads() between
 * levels. At each level, threads read values that other threads may not have
 * written yet (read-after-write hazard), and write values others may still
 * be reading (write-after-read). Whether the result comes out wrong depends
 * on timing, so the bug can pass tests for months.
 *
 * Find it:
 *   compute-sanitizer --tool racecheck ./01_shared_memory_race --buggy
 * racecheck reports the hazard type, the two conflicting accesses, and their
 * source lines.
 *
 * Without --buggy the fixed kernel runs. ctest runs the buggy version under
 * racecheck and passes only if racecheck reports the hazard.
 */
#include <vector>

#include "lab.cuh"

constexpr int THREADS = 256;

__global__ void block_sum(const float *in, float *out, bool buggy) {
  __shared__ float s[THREADS];
  const int tid = threadIdx.x;
  s[tid] = in[blockIdx.x * THREADS + tid];
  __syncthreads();
  for (int stride = THREADS / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s[tid] += s[tid + stride];
    }
    if (!buggy) {
      __syncthreads();  // The fix: finish each level before the next starts.
    }
  }
  if (tid == 0) {
    out[blockIdx.x] = s[0];
  }
}

int main(int argc, char **argv) {
  lab::Args args(argc, argv);
  const bool buggy = args.has("--buggy");
  const int blocks = 64;

  printf("Running the %s kernel%s\n", buggy ? "BUGGY" : "fixed",
         buggy ? "; run under compute-sanitizer --tool racecheck to see the hazard" : "");
  std::vector<float> h_in(blocks * THREADS, 1.0f), h_out(blocks);
  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, h_in.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, blocks * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  block_sum<<<blocks, THREADS>>>(d_in, d_out, buggy);
  CUDA_CHECK_LAUNCH();
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, blocks * sizeof(float), cudaMemcpyDeviceToHost));

  const std::vector<float> expected(blocks, static_cast<float>(THREADS));
  const bool pass = lab::check_equal("block sums", h_out, expected);
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return lab::finish(pass);
}
