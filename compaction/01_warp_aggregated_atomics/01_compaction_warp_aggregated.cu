/*
 * Stream Compaction 1: Warp-Aggregated Atomics (Unstable)
 *
 * Intention:
 * If the output order does not matter, compaction needs no scan: each
 * selected element can reserve an output slot with atomicAdd on a global
 * counter. One atomic per element would serialize badly, so each warp makes
 * a single reservation for all of its selected elements.
 *
 * High-Level Algorithm (per warp, per 32 elements):
 * - mask = __ballot_sync(FULL, keep(x)): bit i is set if lane i keeps its
 *   element.
 * - The lowest selected lane reserves __popc(mask) slots with one atomicAdd
 *   and broadcasts the first slot with __shfl_sync.
 * - Each selected lane writes at base + __popc(mask & lanes_below_me), its
 *   rank among the selected lanes.
 * - Grid-stride loop, so a fixed number of warps covers any input size.
 *
 * Elements stay in order within one warp's 32, but warps reserve slots in
 * whatever order their atomics land, so the output is not in input order.
 * This pattern (also used by the compiler when it aggregates atomics
 * automatically) is often the fastest when order does not matter.
 */
#include "../compaction_harness.cuh"

constexpr int THREADS = 256;
constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

__global__ void compact_warp_aggregated(const int *in, int *out, int *count,
                                        int n) {
  const int lane = threadIdx.x % 32;
  const unsigned lanes_below = (1u << lane) - 1u;
  // Round the loop bound up to a whole warp so all 32 lanes take part in the
  // ballot even past the end of the array.
  const int stride = blockDim.x * gridDim.x;
  for (int base = blockIdx.x * blockDim.x; base < n; base += stride) {
    const int i = base + threadIdx.x;
    const bool valid = i < n;
    const int x = valid ? in[i] : 0;
    const bool selected = valid && keep(x);

    const unsigned mask = __ballot_sync(FULL_MASK, selected);
    if (mask == 0) {
      continue;  // Uniform across the warp: every lane sees the same mask.
    }
    const int leader = __ffs(mask) - 1;
    int slot = 0;
    if (lane == leader) {
      slot = atomicAdd(count, __popc(mask));
    }
    slot = __shfl_sync(FULL_MASK, slot, leader);
    if (selected) {
      out[slot + __popc(mask & lanes_below)] = x;
    }
  }
}

void launch(const int *d_in, int *d_out, int *d_count, int n, void *, size_t) {
  int device = 0;
  int sms = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  const int needed = lab::ceil_div(n, THREADS);
  const int blocks = needed < sms * 8 ? needed : sms * 8;
  CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(int)));
  compact_warp_aggregated<<<blocks, THREADS>>>(d_in, d_out, d_count, n);
}

int main(int argc, char **argv) {
  return run_compaction("1. Warp-aggregated atomics", argc, argv, launch,
                        /*stable=*/false);
}
