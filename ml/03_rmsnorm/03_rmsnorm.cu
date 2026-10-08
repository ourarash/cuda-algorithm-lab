/*
 * RMSNorm
 *
 * Intention:
 * y = x / sqrt(mean(x^2) + eps) * gamma. RMSNorm (Zhang and Sennrich, 2019)
 * drops LayerNorm's mean subtraction and bias; most recent LLMs (LLaMA,
 * Mistral, ...) use it because it is cheaper and works as well.
 *
 * High-Level Algorithm (one 256-thread block per row):
 * 1. Each thread sums x^2 over its elements (float4 loads).
 * 2. Warp shuffles and shared memory reduce to the row's sum of squares.
 * 3. Second pass: scale and write with float4.
 * Only one statistic instead of two, so the reduction is a plain sum; the
 * kernel is bandwidth-bound like LayerNorm.
 */
#include "../rowwise_harness.cuh"

constexpr int THREADS = 256;
constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o /= 2) v += __shfl_xor_sync(FULL_MASK, v, o);
  return v;
}

__device__ __forceinline__ float block_sum(float v) {
  __shared__ float partial[THREADS / 32];
  v = warp_sum(v);
  if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = v;
  __syncthreads();
  v = threadIdx.x % 32 < THREADS / 32 ? partial[threadIdx.x % 32] : 0.0f;
  return warp_sum(v);
}

__global__ void rmsnorm(const float *in, float *out, int cols,
                        const float *gamma) {
  const float4 *x = reinterpret_cast<const float4 *>(in + static_cast<size_t>(blockIdx.x) * cols);
  float4 *y = reinterpret_cast<float4 *>(out + static_cast<size_t>(blockIdx.x) * cols);
  const float4 *g = reinterpret_cast<const float4 *>(gamma);
  const int cols4 = cols / 4;

  float ss = 0.0f;
  for (int c = threadIdx.x; c < cols4; c += THREADS) {
    const float4 v = x[c];
    ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }
  ss = block_sum(ss);
  const float inv_rms = rsqrtf(ss / cols + NORM_EPS);

  for (int c = threadIdx.x; c < cols4; c += THREADS) {
    const float4 v = x[c];
    const float4 gv = g[c];
    y[c] = make_float4(v.x * inv_rms * gv.x, v.y * inv_rms * gv.y,
                       v.z * inv_rms * gv.z, v.w * inv_rms * gv.w);
  }
}

void launch(const float *d_in, float *d_out, int rows, int cols,
            const float *d_gamma, const float *) {
  rmsnorm<<<rows, THREADS>>>(d_in, d_out, cols, d_gamma);
}

int main(int argc, char **argv) {
  return run_rowwise("RMSNorm", argc, argv, launch, rmsnorm_reference,
                     /*rtol=*/1e-4, /*atol=*/1e-5);
}
