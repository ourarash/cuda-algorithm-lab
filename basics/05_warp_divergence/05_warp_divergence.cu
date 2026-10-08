/*
 * Warp Divergence
 *
 * Intention:
 * The 32 threads of a warp share one instruction stream. When they disagree
 * on a branch, the warp runs both sides one after the other, with the lanes
 * that did not take the current side switched off. Per warp, an if/else
 * then costs the time of the if *plus* the time of the else.
 *
 * High-Level Algorithm:
 * Both kernels give half of all threads the expensive path A and the other
 * half the expensive path B, so they do the same amount of work:
 * - divergent: the choice is threadIdx.x % 2, so every warp contains both
 *   kinds of threads and executes both paths;
 * - uniform:   the choice is (threadIdx.x / 32) % 2, so all 32 threads of a
 *   warp agree and each warp executes only one path.
 * The divergent kernel should take about twice as long. Results must match.
 *
 * The fix in real code is usually to reorganize which thread gets which work
 * (sort or bin work items so that neighbouring threads take the same path),
 * as reduction/03_sequential_addressing does compared with step 01.
 */
#include <vector>

#include "lab.cuh"

constexpr int THREADS = 256;
constexpr int STEPS = 256;

__device__ __forceinline__ float path_a(float x) {
  for (int k = 0; k < STEPS; ++k) x = sinf(x) + 0.5f;
  return x;
}
__device__ __forceinline__ float path_b(float x) {
  for (int k = 0; k < STEPS; ++k) x = cosf(x) - 0.25f;
  return x;
}

// `which` selects the path for element i; both kernels use the same mapping
// of elements to paths, only the mapping of paths to lanes differs.
__global__ void divergent(const float *in, float *out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = (threadIdx.x % 2 == 0) ? path_a(in[i]) : path_b(in[i]);
}

__global__ void uniform(const float *in, float *out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = ((threadIdx.x / 32) % 2 == 0) ? path_a(in[i]) : path_b(in[i]);
}

int main(int argc, char **argv) {
  lab::Args args(argc, argv);
  const int n = static_cast<int>(args.get_int("n", args.quick() ? 65536 : 1 << 22));

  lab::print_device();
  const std::vector<float> h_in = lab::random_uniform<float>(n, 0.f, 1.f, 91);
  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  const int blocks = lab::ceil_div(n, THREADS);

  // Each kernel's output, checked against the path its mapping chooses.
  std::vector<float> got(n);
  bool pass = true;
  for (int version = 0; version < 2; ++version) {
    got.resize(n);
    if (version == 0) {
      divergent<<<blocks, THREADS>>>(d_in, d_out, n);
    } else {
      uniform<<<blocks, THREADS>>>(d_in, d_out, n);
    }
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaMemcpy(got.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    // The CPU reference is slow (256 sin/cos per element), so check the first
    // 64K elements: they already contain every lane and warp pattern.
    const int checked = std::min(n, 65536);
    got.resize(checked);
    std::vector<double> expected(checked);
    for (int i = 0; i < checked; ++i) {
      const int lane_id = i % THREADS;
      const bool take_a = version == 0 ? lane_id % 2 == 0 : (lane_id / 32) % 2 == 0;
      double x = h_in[i];
      for (int k = 0; k < STEPS; ++k) x = take_a ? std::sin(x) + 0.5 : std::cos(x) - 0.25;
      expected[i] = x;
    }
    // sinf/cosf differ from the double CPU math in the last bits each step.
    pass = lab::check_close(version == 0 ? "divergent" : "uniform", got, expected,
                            1e-3, 1e-4) && pass;
  }

  const float div_ms = lab::time_ms([&] { divergent<<<blocks, THREADS>>>(d_in, d_out, n); });
  const float uni_ms = lab::time_ms([&] { uniform<<<blocks, THREADS>>>(d_in, d_out, n); });
  lab::report("divergent (lane % 2)", div_ms, 0, 0);
  lab::report("uniform (warp % 2)", uni_ms, 0, 0);
  printf("%-30s %9.2f x\n", "Divergence slowdown", div_ms / uni_ms);

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return lab::finish(pass);
}
