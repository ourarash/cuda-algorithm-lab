/*
 * Stream Compaction 0: Scan-Based (Stable)
 *
 * Intention:
 * The textbook way to compact in parallel: an element's output position is
 * the number of selected elements before it, which is an exclusive scan of
 * the 0/1 "keep" flags. Every thread then knows exactly where to write, with
 * no atomics, and the output keeps the input order.
 *
 * High-Level Algorithm (the reduce-then-scan structure of
 * scan/07_warp_shuffle_reduce_then_scan, applied to keep flags):
 * 1. count_tiles: each 1024-element tile counts how many it keeps.
 * 2. scan_tile_counts: one block turns the counts into exclusive tile offsets
 *    and writes the total, which is the output size.
 * 3. scatter_tiles: each tile scans its flags (thread-local, then warp
 *    shuffles, then across warps) and every selected element is written to
 *    tile_offset + (number of selected elements before it in the tile).
 */
#include "../compaction_harness.cuh"

constexpr int BLOCK = 256;
constexpr int ITEMS = 4;
constexpr int TILE = BLOCK * ITEMS;
constexpr int WARPS = BLOCK / 32;
constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

__device__ __forceinline__ int warp_inclusive_scan(int v) {
  const int lane = threadIdx.x % 32;
#pragma unroll
  for (int offset = 1; offset < 32; offset *= 2) {
    const int up = __shfl_up_sync(FULL_MASK, v, offset);
    if (lane >= offset) {
      v += up;
    }
  }
  return v;
}

__device__ __forceinline__ int block_exclusive_scan(int v, int &block_total) {
  __shared__ int warp_sums[WARPS];
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int inclusive = warp_inclusive_scan(v);
  if (lane == 31) {
    warp_sums[warp] = inclusive;
  }
  __syncthreads();
  if (warp == 0) {
    int s = lane < WARPS ? warp_sums[lane] : 0;
    s = warp_inclusive_scan(s);
    if (lane < WARPS) {
      warp_sums[lane] = s;
    }
  }
  __syncthreads();
  const int warp_prefix = warp == 0 ? 0 : warp_sums[warp - 1];
  block_total = warp_sums[WARPS - 1];
  __syncthreads();
  return warp_prefix + inclusive - v;
}

// Loads this thread's 4 consecutive elements; out-of-range slots are invalid.
__device__ __forceinline__ void load_items(const int *in, int n, int base,
                                           int (&items)[ITEMS],
                                           bool (&valid)[ITEMS]) {
  const int first = base + threadIdx.x * ITEMS;
#pragma unroll
  for (int i = 0; i < ITEMS; ++i) {
    valid[i] = first + i < n;
    items[i] = valid[i] ? in[first + i] : 0;
  }
}

__global__ void count_tiles(const int *in, int n, int *tile_counts) {
  int items[ITEMS];
  bool valid[ITEMS];
  load_items(in, n, blockIdx.x * TILE, items, valid);
  int kept = 0;
#pragma unroll
  for (int i = 0; i < ITEMS; ++i) {
    kept += valid[i] && keep(items[i]);
  }
  int tile_total;
  block_exclusive_scan(kept, tile_total);
  if (threadIdx.x == 0) {
    tile_counts[blockIdx.x] = tile_total;
  }
}

// Exclusive scan of the tile counts by one block, 1024 at a time; writes the
// grand total to *count.
__global__ void scan_tile_counts(int *tile_counts, int num_tiles, int *count) {
  int carry = 0;
  for (int base = 0; base < num_tiles; base += TILE) {
    const int first = base + threadIdx.x * ITEMS;
    int items[ITEMS];
    int thread_sum = 0;
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
      items[i] = first + i < num_tiles ? tile_counts[first + i] : 0;
      thread_sum += items[i];
    }
    int chunk_total;
    int running = carry + block_exclusive_scan(thread_sum, chunk_total);
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
      if (first + i < num_tiles) {
        tile_counts[first + i] = running;  // Exclusive offset
      }
      running += items[i];
    }
    carry += chunk_total;
  }
  if (threadIdx.x == 0) {
    *count = carry;
  }
}

__global__ void scatter_tiles(const int *in, int n, const int *tile_offsets,
                              int *out) {
  int items[ITEMS];
  bool valid[ITEMS];
  load_items(in, n, blockIdx.x * TILE, items, valid);
  int kept = 0;
#pragma unroll
  for (int i = 0; i < ITEMS; ++i) {
    kept += valid[i] && keep(items[i]);
  }
  int tile_total;
  int position = tile_offsets[blockIdx.x] + block_exclusive_scan(kept, tile_total);
#pragma unroll
  for (int i = 0; i < ITEMS; ++i) {
    if (valid[i] && keep(items[i])) {
      out[position++] = items[i];
    }
  }
}

void launch(const int *d_in, int *d_out, int *d_count, int n, void *d_scratch,
            size_t) {
  const int num_tiles = lab::ceil_div(n, TILE);
  int *tile_counts = static_cast<int *>(d_scratch);
  count_tiles<<<num_tiles, BLOCK>>>(d_in, n, tile_counts);
  scan_tile_counts<<<1, BLOCK>>>(tile_counts, num_tiles, d_count);
  scatter_tiles<<<num_tiles, BLOCK>>>(d_in, n, tile_counts, d_out);
}

int main(int argc, char **argv) {
  return run_compaction("0. Scan-based", argc, argv, launch, /*stable=*/true);
}
