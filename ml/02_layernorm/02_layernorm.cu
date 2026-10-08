/*
 * LayerNorm
 *
 * Intention:
 * y = (x - mean) / sqrt(var + eps) * gamma + beta, per row, with learned
 * per-column gamma and beta. Every transformer block applies it (or RMSNorm,
 * step 03) twice per token.
 *
 * Numerics: the textbook var = E[x^2] - E[x]^2 loses precision when the mean
 * is large relative to the spread. Welford's algorithm updates a running
 * (count, mean, M2) instead, and two partial results merge exactly:
 *   delta = mean_b - mean_a,  n = n_a + n_b
 *   mean  = mean_a + delta * n_b / n
 *   M2    = M2_a + M2_b + delta^2 * n_a * n_b / n
 * with var = M2 / n at the end.
 *
 * High-Level Algorithm (one 256-thread block per row):
 * 1. Each thread runs Welford over its elements (float4 loads).
 * 2. Warp shuffles and shared memory merge the 256 partial statistics.
 * 3. Second pass: normalize, scale, shift, and write with float4.
 */
#include "../rowwise_harness.cuh"

constexpr int THREADS = 256;
constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

struct Welford {
  float n;
  float mean;
  float m2;
};

__device__ __forceinline__ void add(Welford &w, float x) {
  w.n += 1.0f;
  const float delta = x - w.mean;
  w.mean += delta / w.n;
  w.m2 += delta * (x - w.mean);
}

__device__ __forceinline__ Welford merge(Welford a, Welford b) {
  const float n = a.n + b.n;
  if (n == 0.0f) return a;
  const float delta = b.mean - a.mean;
  const float nb_over_n = b.n / n;
  return {n, a.mean + delta * nb_over_n, a.m2 + b.m2 + delta * delta * a.n * nb_over_n};
}

__device__ __forceinline__ Welford warp_merge(Welford w) {
#pragma unroll
  for (int o = 16; o > 0; o /= 2) {
    Welford other = {__shfl_xor_sync(FULL_MASK, w.n, o),
                     __shfl_xor_sync(FULL_MASK, w.mean, o),
                     __shfl_xor_sync(FULL_MASK, w.m2, o)};
    w = merge(w, other);
  }
  return w;
}

__device__ __forceinline__ Welford block_merge(Welford w) {
  __shared__ Welford partial[THREADS / 32];
  w = warp_merge(w);
  if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = w;
  __syncthreads();
  w = threadIdx.x % 32 < THREADS / 32 ? partial[threadIdx.x % 32]
                                      : Welford{0.0f, 0.0f, 0.0f};
  return warp_merge(w);
}

__global__ void layernorm(const float *in, float *out, int cols,
                          const float *gamma, const float *beta) {
  const float4 *x = reinterpret_cast<const float4 *>(in + static_cast<size_t>(blockIdx.x) * cols);
  float4 *y = reinterpret_cast<float4 *>(out + static_cast<size_t>(blockIdx.x) * cols);
  const float4 *g = reinterpret_cast<const float4 *>(gamma);
  const float4 *b = reinterpret_cast<const float4 *>(beta);
  const int cols4 = cols / 4;

  Welford w = {0.0f, 0.0f, 0.0f};
  for (int c = threadIdx.x; c < cols4; c += THREADS) {
    const float4 v = x[c];
    add(w, v.x);
    add(w, v.y);
    add(w, v.z);
    add(w, v.w);
  }
  w = block_merge(w);
  const float mean = w.mean;
  const float inv_std = rsqrtf(w.m2 / cols + NORM_EPS);

  for (int c = threadIdx.x; c < cols4; c += THREADS) {
    const float4 v = x[c];
    const float4 gv = g[c];
    const float4 bv = b[c];
    y[c] = make_float4((v.x - mean) * inv_std * gv.x + bv.x,
                       (v.y - mean) * inv_std * gv.y + bv.y,
                       (v.z - mean) * inv_std * gv.z + bv.z,
                       (v.w - mean) * inv_std * gv.w + bv.w);
  }
}

void launch(const float *d_in, float *d_out, int rows, int cols,
            const float *d_gamma, const float *d_beta) {
  layernorm<<<rows, THREADS>>>(d_in, d_out, cols, d_gamma, d_beta);
}

int main(int argc, char **argv) {
  return run_rowwise("LayerNorm (Welford)", argc, argv, launch,
                     layernorm_reference,
                     /*rtol=*/1e-4, /*atol=*/1e-5);
}
