/*
 * Attention 1: FlashAttention (Forward Pass)
 *
 * Intention:
 * FlashAttention (Dao et al., 2022) computes exactly the same output as
 * step 04 without ever writing the n x n score matrix to memory. It tiles
 * the computation so scores live only in registers, and uses the online
 * softmax update from ml/01_softmax_online to combine tiles of keys.
 *
 * High-Level Algorithm (this minimal version):
 * - One block per 64 query rows of one head; thread t owns query row t: its
 *   64 outputs (acc), its running max m, and its running sum l.
 * - The block's Q tile is loaded into shared memory once.
 * - Loop over the keys in tiles of 32:
 *     load the K and V tiles into shared memory;
 *     each thread computes its 32 scores s_j = q . k_j * scale (registers);
 *     m_new = max(m, max_j s_j); rescale acc and l by exp(m - m_new);
 *     add exp(s_j - m_new) to l and exp(s_j - m_new) * v_j to acc.
 * - O = acc / l, staged through shared memory so the global writes are
 *   coalesced.
 * Global traffic is reading Q once and K and V once per block of queries:
 * the n x n matrix of step 04 is gone, and memory use is linear in n.
 *
 * Shared-memory details: thread t reads row t of the Q tile while the other
 * threads of its warp read rows t + 1, t + 2, ... at the same column, so the
 * Q tile rows are padded to 65 floats to put them in different banks. All
 * threads read the same K and V element at the same time, which is a
 * broadcast and needs no padding.
 *
 * The real FlashAttention kernels assign a warp per row group, use Tensor
 * Cores for both matrix products, and add causal masking and a backward
 * pass; this version keeps only the idea that makes them work.
 */
#include <cfloat>

#include "../attention_harness.cuh"

constexpr int BR = 64;  // Query rows per block = threads per block
constexpr int BC = 32;  // Keys per tile

__global__ void __launch_bounds__(BR)
    flash_attention(const float *q, const float *k, const float *v, float *o,
                    int n, float scale) {
  __shared__ float Qs[BR][HEAD_DIM + 1];
  __shared__ float Ks[BC][HEAD_DIM];
  __shared__ float Vs[BC][HEAD_DIM];

  const int t = threadIdx.x;
  const int row0 = blockIdx.x * BR;
  const size_t head = static_cast<size_t>(blockIdx.y) * n * HEAD_DIM;

  // Coalesced load of the Q tile (rows past the end are zero).
  for (int idx = t; idx < BR * HEAD_DIM; idx += BR) {
    const int r = idx / HEAD_DIM;
    const int c = idx % HEAD_DIM;
    Qs[r][c] = row0 + r < n ? q[head + static_cast<size_t>(row0 + r) * HEAD_DIM + c] : 0.0f;
  }

  float acc[HEAD_DIM];
#pragma unroll
  for (int c = 0; c < HEAD_DIM; ++c) acc[c] = 0.0f;
  float m = -FLT_MAX;  // Running max of this row's scores
  float l = 0.0f;      // Running sum of exp(score - m)

  for (int j0 = 0; j0 < n; j0 += BC) {
    __syncthreads();  // Previous K/V tile fully used (and Q tile loaded)
    for (int idx = t; idx < BC * HEAD_DIM; idx += BR) {
      const int r = idx / HEAD_DIM;
      const int c = idx % HEAD_DIM;
      const bool valid = j0 + r < n;
      const size_t g = head + static_cast<size_t>(j0 + r) * HEAD_DIM + c;
      Ks[r][c] = valid ? k[g] : 0.0f;
      Vs[r][c] = valid ? v[g] : 0.0f;
    }
    __syncthreads();

    // Scores for this tile, kept in registers. Keys past the end get -inf
    // in effect (exp underflows to exactly 0).
    float s[BC];
    float tile_max = -FLT_MAX;
#pragma unroll
    for (int jj = 0; jj < BC; ++jj) {
      float dot = 0.0f;
#pragma unroll
      for (int c = 0; c < HEAD_DIM; ++c) dot += Qs[t][c] * Ks[jj][c];
      s[jj] = j0 + jj < n ? dot * scale : -FLT_MAX;
      tile_max = fmaxf(tile_max, s[jj]);
    }

    // Online softmax: rescale what was accumulated under the old max.
    const float m_new = fmaxf(m, tile_max);
    const float correction = __expf(m - m_new);
    l *= correction;
#pragma unroll
    for (int c = 0; c < HEAD_DIM; ++c) acc[c] *= correction;

#pragma unroll
    for (int jj = 0; jj < BC; ++jj) {
      const float p = __expf(s[jj] - m_new);
      l += p;
#pragma unroll
      for (int c = 0; c < HEAD_DIM; ++c) acc[c] += p * Vs[jj][c];
    }
    m = m_new;
  }

  // Stage the normalized output rows in shared memory (reusing the Q tile),
  // then write them out coalesced.
  __syncthreads();
  const float inv_l = 1.0f / l;
#pragma unroll
  for (int c = 0; c < HEAD_DIM; ++c) Qs[t][c] = acc[c] * inv_l;
  __syncthreads();
  for (int idx = t; idx < BR * HEAD_DIM; idx += BR) {
    const int r = idx / HEAD_DIM;
    const int c = idx % HEAD_DIM;
    if (row0 + r < n) {
      o[head + static_cast<size_t>(row0 + r) * HEAD_DIM + c] = Qs[r][c];
    }
  }
}

void launch(const float *q, const float *k, const float *v, float *o, int bh,
            int n, float *) {
  const float scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));
  dim3 grid(lab::ceil_div(n, BR), bh);
  flash_attention<<<grid, BR>>>(q, k, v, o, n, scale);
}

int main(int argc, char **argv) {
  return run_attention("FlashAttention forward", argc, argv, launch);
}
