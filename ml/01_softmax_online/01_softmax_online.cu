/*
 * Softmax 1: Online Softmax (Two Passes)
 *
 * Intention:
 * The max and the sum of exponentials can be computed in a single pass
 * (Milakov and Gimelshein, "Online normalizer calculation for softmax",
 * 2018). Keep a running max m and a running sum d of exp(x - m). When a new
 * value raises the max, rescale the sum so far:
 *   m_new = max(m, x)
 *   d     = d * exp(m - m_new) + exp(x - m_new)
 * Two partial results (m1, d1) and (m2, d2) merge the same way:
 *   m = max(m1, m2),  d = d1 * exp(m1 - m) + d2 * exp(m2 - m)
 * This merge rule is exactly what FlashAttention uses to process attention
 * one block of keys at a time (see 05_flash_attention).
 *
 * High-Level Algorithm (one 256-thread block per row):
 * 1. Each thread streams its elements with float4 loads, updating (m, d).
 * 2. Warp shuffles and shared memory merge the 256 (m, d) pairs.
 * 3. Second pass: write exp(x - m) / d, again with float4.
 * The row is read twice instead of three times.
 */
#include <cfloat>

#include "../rowwise_harness.cuh"

constexpr int THREADS = 256;
constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

struct MaxSum {
  float m;  // Running maximum
  float d;  // Sum of exp(x - m)
};

__device__ __forceinline__ MaxSum merge(MaxSum a, MaxSum b) {
  const float m = fmaxf(a.m, b.m);
  return {m, a.d * __expf(a.m - m) + b.d * __expf(b.m - m)};
}

__device__ __forceinline__ MaxSum warp_merge(MaxSum v) {
#pragma unroll
  for (int o = 16; o > 0; o /= 2) {
    MaxSum other = {__shfl_xor_sync(FULL_MASK, v.m, o),
                    __shfl_xor_sync(FULL_MASK, v.d, o)};
    v = merge(v, other);
  }
  return v;
}

__device__ __forceinline__ MaxSum block_merge(MaxSum v) {
  __shared__ MaxSum partial[THREADS / 32];
  v = warp_merge(v);
  if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = v;
  __syncthreads();
  v = threadIdx.x % 32 < THREADS / 32 ? partial[threadIdx.x % 32]
                                      : MaxSum{-FLT_MAX, 0.0f};
  return warp_merge(v);
}

__device__ __forceinline__ void update(MaxSum &s, float x) {
  if (x > s.m) {
    s.d = s.d * __expf(s.m - x) + 1.0f;
    s.m = x;
  } else {
    s.d += __expf(x - s.m);
  }
}

__global__ void softmax_online(const float *in, float *out, int cols) {
  const float4 *x = reinterpret_cast<const float4 *>(in + static_cast<size_t>(blockIdx.x) * cols);
  float4 *y = reinterpret_cast<float4 *>(out + static_cast<size_t>(blockIdx.x) * cols);
  const int cols4 = cols / 4;

  MaxSum s = {-FLT_MAX, 0.0f};
  for (int c = threadIdx.x; c < cols4; c += THREADS) {
    const float4 v = x[c];
    update(s, v.x);
    update(s, v.y);
    update(s, v.z);
    update(s, v.w);
  }
  s = block_merge(s);

  const float inv = 1.0f / s.d;
  for (int c = threadIdx.x; c < cols4; c += THREADS) {
    const float4 v = x[c];
    y[c] = make_float4(__expf(v.x - s.m) * inv, __expf(v.y - s.m) * inv,
                       __expf(v.z - s.m) * inv, __expf(v.w - s.m) * inv);
  }
}

void launch(const float *d_in, float *d_out, int rows, int cols, const float *,
            const float *) {
  softmax_online<<<rows, THREADS>>>(d_in, d_out, cols);
}

int main(int argc, char **argv) {
  return run_rowwise("1. Online softmax", argc, argv, launch, softmax_reference,
                     /*rtol=*/1e-4, /*atol=*/1e-9);
}
