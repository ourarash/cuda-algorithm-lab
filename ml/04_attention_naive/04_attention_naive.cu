/*
 * Attention 0: Naive (Materialized Score Matrix)
 *
 * Intention:
 * The direct implementation of O = softmax(Q K^T / sqrt(d)) V with three
 * kernels, the way a framework runs it as separate operations:
 * 1. scores:  S = Q K^T * scale, an n x n matrix per head, written to global
 *    memory (one thread per score).
 * 2. softmax: each row of S, in place (one block per row, three passes as in
 *    00_softmax_naive).
 * 3. output:  O = S V (one thread per output element).
 *
 * The cost is the n x n matrix: for 16 heads at n = 1024 it is 64 MB, which
 * is written, read and rewritten by the softmax (three passes), and read again
 * for the output, against 16 MB for Q, K, V, and O together; and it grows
 * quadratically with the sequence length. Step 05 never stores it.
 */
#include <cfloat>

#include "../attention_harness.cuh"

constexpr int THREADS = 256;
constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

__global__ void scores_kernel(const float *q, const float *k, float *s, int n,
                              float scale) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;  // key
  const int i = blockIdx.y * blockDim.y + threadIdx.y;  // query
  const int b = blockIdx.z;
  if (i >= n || j >= n) return;
  const float *qi = q + (static_cast<size_t>(b) * n + i) * HEAD_DIM;
  const float *kj = k + (static_cast<size_t>(b) * n + j) * HEAD_DIM;
  float dot = 0.0f;
  for (int c = 0; c < HEAD_DIM; ++c) dot += qi[c] * kj[c];
  s[(static_cast<size_t>(b) * n + i) * n + j] = dot * scale;
}

__device__ __forceinline__ float block_reduce(float v, bool is_max) {
  __shared__ float partial[THREADS / 32];
  for (int o = 16; o > 0; o /= 2) {
    const float other = __shfl_xor_sync(FULL_MASK, v, o);
    v = is_max ? fmaxf(v, other) : v + other;
  }
  if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = v;
  __syncthreads();
  v = threadIdx.x % 32 < THREADS / 32 ? partial[threadIdx.x % 32]
                                      : (is_max ? -FLT_MAX : 0.0f);
  for (int o = 16; o > 0; o /= 2) {
    const float other = __shfl_xor_sync(FULL_MASK, v, o);
    v = is_max ? fmaxf(v, other) : v + other;
  }
  __syncthreads();
  return v;
}

__global__ void softmax_rows(float *s, int n) {
  float *row = s + static_cast<size_t>(blockIdx.x) * n;
  float m = -FLT_MAX;
  for (int j = threadIdx.x; j < n; j += THREADS) m = fmaxf(m, row[j]);
  m = block_reduce(m, true);
  float sum = 0.0f;
  for (int j = threadIdx.x; j < n; j += THREADS) sum += __expf(row[j] - m);
  sum = block_reduce(sum, false);
  const float inv = 1.0f / sum;
  for (int j = threadIdx.x; j < n; j += THREADS) row[j] = __expf(row[j] - m) * inv;
}

// One block per (query row, head); thread c computes O[i][c].
__global__ void output_kernel(const float *p, const float *v, float *o, int n) {
  const int i = blockIdx.x;
  const int b = blockIdx.y;
  const int c = threadIdx.x;
  const float *pi = p + (static_cast<size_t>(b) * n + i) * n;
  const float *vb = v + static_cast<size_t>(b) * n * HEAD_DIM;
  float acc = 0.0f;
  for (int j = 0; j < n; ++j) acc += pi[j] * vb[j * HEAD_DIM + c];
  o[(static_cast<size_t>(b) * n + i) * HEAD_DIM + c] = acc;
}

void launch(const float *q, const float *k, const float *v, float *o, int bh,
            int n, float *scores) {
  const float scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));
  dim3 block(32, 8);
  dim3 grid(lab::ceil_div(n, 32), lab::ceil_div(n, 8), bh);
  scores_kernel<<<grid, block>>>(q, k, scores, n, scale);
  softmax_rows<<<bh * n, THREADS>>>(scores, n);
  output_kernel<<<dim3(n, bh), HEAD_DIM>>>(scores, v, o, n);
}

int main(int argc, char **argv) {
  return run_attention("Attention, naive (n x n scores)", argc, argv, launch);
}
