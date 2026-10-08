/*
 * Softmax 0: Three Passes
 *
 * Intention:
 * softmax(x)_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x)), for every row.
 * Subtracting the row maximum keeps exp() from overflowing and does not
 * change the result. The direct implementation needs three passes over the
 * row: find the max, sum the exponentials, write the normalized values.
 *
 * High-Level Algorithm (one 256-thread block per row):
 * 1. Each thread takes the max over its strided elements; warp shuffles and
 *    shared memory reduce that to the row max.
 * 2. Each thread sums exp(x - max) over its elements; reduce to the row sum.
 * 3. Each thread writes exp(x - max) / sum for its elements.
 *
 * The row is read from memory three times. For short rows the second and
 * third reads mostly hit in L1/L2; for long rows they cost real bandwidth.
 * Step 01 needs only two reads.
 */
#include <cfloat>

#include "../rowwise_harness.cuh"

constexpr int THREADS = 256;
constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o /= 2) v = fmaxf(v, __shfl_xor_sync(FULL_MASK, v, o));
  return v;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o /= 2) v += __shfl_xor_sync(FULL_MASK, v, o);
  return v;
}

// Block-wide reductions; every thread gets the result. (XOR shuffles leave
// the result in all lanes, not only lane 0.)
__device__ __forceinline__ float block_max(float v) {
  __shared__ float partial[THREADS / 32];
  v = warp_max(v);
  if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = v;
  __syncthreads();
  v = threadIdx.x % 32 < THREADS / 32 ? partial[threadIdx.x % 32] : -FLT_MAX;
  v = warp_max(v);
  __syncthreads();
  return v;
}
__device__ __forceinline__ float block_sum(float v) {
  __shared__ float partial[THREADS / 32];
  v = warp_sum(v);
  if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = v;
  __syncthreads();
  v = threadIdx.x % 32 < THREADS / 32 ? partial[threadIdx.x % 32] : 0.0f;
  v = warp_sum(v);
  __syncthreads();
  return v;
}

__global__ void softmax_three_pass(const float *in, float *out, int cols) {
  const float *x = in + static_cast<size_t>(blockIdx.x) * cols;
  float *y = out + static_cast<size_t>(blockIdx.x) * cols;

  float m = -FLT_MAX;
  for (int c = threadIdx.x; c < cols; c += THREADS) m = fmaxf(m, x[c]);
  m = block_max(m);

  float sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += THREADS) sum += __expf(x[c] - m);
  sum = block_sum(sum);

  const float inv = 1.0f / sum;
  for (int c = threadIdx.x; c < cols; c += THREADS) y[c] = __expf(x[c] - m) * inv;
}

void launch(const float *d_in, float *d_out, int rows, int cols, const float *,
            const float *) {
  softmax_three_pass<<<rows, THREADS>>>(d_in, d_out, cols);
}

int main(int argc, char **argv) {
  return run_rowwise("0. Softmax, three passes", argc, argv, launch,
                     softmax_reference,
                     /*rtol=*/1e-4, /*atol=*/1e-9);
}
