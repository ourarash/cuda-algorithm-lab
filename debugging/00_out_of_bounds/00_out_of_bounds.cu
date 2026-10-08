/*
 * Debugging 0: Out-of-Bounds Access (compute-sanitizer memcheck)
 *
 * The bug: an off-by-one bounds check, `i <= n` instead of `i < n`. The
 * thread with i == n writes one element past the end of the allocation. On
 * real hardware this often does not crash (the allocation is rounded up), so
 * the program may print correct results and the bug stays hidden until it
 * corrupts a neighbouring buffer.
 *
 * Find it:
 *   compute-sanitizer --tool memcheck ./00_out_of_bounds --buggy
 * memcheck reports "Invalid __global__ write of size 4", the kernel, the
 * thread and block, and (with -lineinfo, which the build adds) the source
 * line.
 *
 * Without --buggy the fixed kernel runs. ctest runs the buggy version under
 * memcheck and passes only if memcheck reports the error.
 */
#include <vector>

#include "lab.cuh"

__global__ void scale_buggy(const float *in, float *out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i <= n) {  // BUG: lets i == n through
    out[i] = 2.0f * in[i];
  }
}

__global__ void scale_fixed(const float *in, float *out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = 2.0f * in[i];
  }
}

int main(int argc, char **argv) {
  lab::Args args(argc, argv);
  const bool buggy = args.has("--buggy");
  const int n = 1000;  // Not a multiple of the block size, so thread n exists

  printf("Running the %s kernel%s\n", buggy ? "BUGGY" : "fixed",
         buggy ? "; run under compute-sanitizer --tool memcheck to see the error" : "");
  std::vector<float> h_in(n, 1.5f), h_out(n);
  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  if (buggy) {
    scale_buggy<<<lab::ceil_div(n, 256), 256>>>(d_in, d_out, n);
  } else {
    scale_fixed<<<lab::ceil_div(n, 256), 256>>>(d_in, d_out, n);
  }
  CUDA_CHECK_LAUNCH();
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

  const std::vector<float> expected(n, 3.0f);
  const bool pass = lab::check_equal("output", h_out, expected);
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return lab::finish(pass);
}
