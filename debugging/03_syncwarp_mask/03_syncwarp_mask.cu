/*
 * Debugging 3: Wrong __syncwarp() Mask (compute-sanitizer synccheck)
 *
 * The bug: __syncwarp(mask) synchronizes the lanes named in `mask`, and
 * every lane that calls it must be in the mask. Here a warp-level reduction
 * in shared memory (like reduction/05_unroll_last_warp) is executed by all 32
 * lanes, but synchronizes with __syncwarp(0x0000FFFF), a mask that only
 * names lanes 0-15, perhaps copied from code where only half a warp took
 * part. For lanes 16-31 the behaviour is undefined, and the reduction may
 * read values before they are written.
 *
 * Find it:
 *   compute-sanitizer --tool synccheck ./03_syncwarp_mask --buggy
 * synccheck reports "Invalid arguments" for the barrier: threads reached a
 * __syncwarp() whose mask does not include them.
 *
 * synccheck also detects __syncthreads() inside a branch that not all threads
 * of a block take, on architectures where that is not supported. The fix
 * here is the full mask, __syncwarp() (equivalent to 0xFFFFFFFF), because all
 * 32 lanes participate. Without --buggy the fixed kernel runs; ctest runs the
 * buggy version under synccheck and passes only if synccheck reports the
 * error.
 */
#include <vector>

#include "lab.cuh"

constexpr int WARPS = 8;

__global__ void warp_sums(const float *in, float *out, unsigned mask) {
  __shared__ float s[WARPS][32];
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  float v = in[blockIdx.x * blockDim.x + threadIdx.x];
  s[warp][lane] = v;
  __syncwarp(mask);
  for (int offset = 16; offset > 0; offset /= 2) {
    if (lane < offset) {
      v += s[warp][lane + offset];
    }
    __syncwarp(mask);
    s[warp][lane] = v;
    __syncwarp(mask);
  }
  if (lane == 0) {
    out[blockIdx.x * WARPS + warp] = s[warp][0];
  }
}

int main(int argc, char **argv) {
  lab::Args args(argc, argv);
  const bool buggy = args.has("--buggy");
  const int blocks = 16;
  const int n = blocks * WARPS * 32;

  printf("Running the %s kernel%s\n", buggy ? "BUGGY" : "fixed",
         buggy ? "; run under compute-sanitizer --tool synccheck to see the error" : "");
  std::vector<float> h_in(n), h_out(blocks * WARPS), expected(blocks * WARPS, 0.0f);
  for (int i = 0; i < n; ++i) {
    h_in[i] = static_cast<float>(i % 7);
    expected[i / 32] += h_in[i];
  }
  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, h_out.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  const unsigned mask = buggy ? 0x0000FFFFu : 0xFFFFFFFFu;
  warp_sums<<<blocks, WARPS * 32>>>(d_in, d_out, mask);
  CUDA_CHECK_LAUNCH();
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  const bool pass = lab::check_equal("warp sums", h_out, expected);
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return lab::finish(pass);
}
