/*
 * Debugging 2: Reading Uninitialized Memory (compute-sanitizer initcheck)
 *
 * The bug: cudaMalloc does not zero memory, but the kernel accumulates into
 * its output (out[i] += ...) as if it did. If the allocation happens to reuse
 * memory that was zero, the result is right; otherwise it is garbage. That
 * makes the bug depend on what ran before.
 *
 * Find it:
 *   compute-sanitizer --tool initcheck ./02_uninitialized_memory --buggy
 * initcheck reports "Uninitialized __global__ memory read" at the += line.
 *
 * The fix is to clear the output first (cudaMemset), or to write it with =
 * on the first contribution. Without --buggy the fixed version runs; ctest
 * runs the buggy version under initcheck and passes only if initcheck
 * reports the error.
 */
#include <vector>

#include "lab.cuh"

__global__ void accumulate(const float *a, const float *b, float *out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] += a[i];  // Reads out[i] before anything wrote it, unless cleared
    out[i] += b[i];
  }
}

int main(int argc, char **argv) {
  lab::Args args(argc, argv);
  const bool buggy = args.has("--buggy");
  const int n = 4096;

  printf("Running the %s version%s\n", buggy ? "BUGGY" : "fixed",
         buggy ? "; run under compute-sanitizer --tool initcheck to see the error" : "");
  std::vector<float> h_a(n, 1.0f), h_b(n, 2.0f), h_out(n);
  float *d_a, *d_b, *d_out;
  CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  if (!buggy) {
    CUDA_CHECK(cudaMemset(d_out, 0, n * sizeof(float)));  // The fix
  }

  accumulate<<<lab::ceil_div(n, 256), 256>>>(d_a, d_b, d_out, n);
  CUDA_CHECK_LAUNCH();
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

  const std::vector<float> expected(n, 3.0f);
  const bool pass = lab::check_equal("output", h_out, expected);
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_out));
  return lab::finish(pass);
}
