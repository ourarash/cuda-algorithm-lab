/*
 * Large-Array Scan with Warp Shuffles (Reduce-Then-Scan)
 *
 * Intention:
 * Steps 00-06 scan in shared memory with one element per thread and need
 * O(log n) levels of recursion for large arrays. This step uses the modern
 * building blocks instead: each thread scans several elements in registers,
 * warps scan with shuffles, and a large array takes exactly three kernel
 * launches regardless of size.
 *
 * Tile scan (shared by all three kernels):
 * - A block of 256 threads owns a tile of 1024 ints; thread t holds the 4
 *   consecutive elements 4t .. 4t + 3, loaded with one 16-byte int4 load.
 * - Each thread scans its 4 items sequentially in registers.
 * - warp_inclusive_scan: __shfl_up_sync with offsets 1, 2, 4, 8, 16 gives
 *   every lane the sum of its own and all lower lanes' totals.
 * - The 8 warp totals go through shared memory and are scanned by warp 0 the
 *   same way; each warp then adds the totals of the warps before it.
 *
 * Large arrays (reduce-then-scan):
 * 1. reduce_tiles: every tile writes its total.
 * 2. scan_tile_sums: one block turns the tile totals into exclusive offsets
 *    (the sum of all tiles before each tile), walking them 1024 at a time.
 * 3. scan_tiles: every tile scans itself again and adds its offset.
 * The input is read twice (passes 1 and 3), so the minimum is 3n memory
 * traffic instead of 2n. Step 08 removes the second read.
 */
#include "../scan_harness.cuh"

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

// Exclusive scan of one value per thread across the block. Also returns the
// block's total in every thread.
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
  __syncthreads();  // warp_sums is reused by the next call
  return warp_prefix + inclusive - v;
}

__device__ __forceinline__ void load_items(const int *in, int n, int base,
                                           int (&items)[ITEMS]) {
  const int first = base + threadIdx.x * ITEMS;
  if (first + ITEMS <= n) {
    // base is a multiple of TILE, so `first` is a multiple of 4: aligned.
    const int4 v = *reinterpret_cast<const int4 *>(in + first);
    items[0] = v.x;
    items[1] = v.y;
    items[2] = v.z;
    items[3] = v.w;
  } else {
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
      items[i] = first + i < n ? in[first + i] : 0;
    }
  }
}

__device__ __forceinline__ void store_items(int *out, int n, int base,
                                            const int (&items)[ITEMS]) {
  const int first = base + threadIdx.x * ITEMS;
  if (first + ITEMS <= n) {
    *reinterpret_cast<int4 *>(out + first) =
        make_int4(items[0], items[1], items[2], items[3]);
  } else {
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
      if (first + i < n) {
        out[first + i] = items[i];
      }
    }
  }
}

// Scans the tile in place (block-local inclusive scan) and returns its total.
__device__ __forceinline__ int scan_tile(int (&items)[ITEMS]) {
  int thread_sum = 0;
#pragma unroll
  for (int i = 0; i < ITEMS; ++i) {
    thread_sum += items[i];
    items[i] = thread_sum;
  }
  int tile_total;
  const int thread_prefix = block_exclusive_scan(thread_sum, tile_total);
#pragma unroll
  for (int i = 0; i < ITEMS; ++i) {
    items[i] += thread_prefix;
  }
  return tile_total;
}

// Pass 1: one total per tile.
__global__ void reduce_tiles(const int *in, int *tile_sums, int n) {
  int items[ITEMS];
  load_items(in, n, blockIdx.x * TILE, items);
  const int total = scan_tile(items);
  if (threadIdx.x == 0) {
    tile_sums[blockIdx.x] = total;
  }
}

// Pass 2: tile totals -> exclusive tile offsets, in place, by a single block.
__global__ void scan_tile_sums(int *tile_sums, int num_tiles) {
  int carry = 0;
  for (int base = 0; base < num_tiles; base += TILE) {
    int items[ITEMS];
    int original[ITEMS];
    load_items(tile_sums, num_tiles, base, items);
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
      original[i] = items[i];
    }
    const int chunk_total = scan_tile(items);
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
      items[i] += carry - original[i];  // Inclusive -> exclusive, plus carry
    }
    store_items(tile_sums, num_tiles, base, items);
    carry += chunk_total;
  }
}

// Pass 3: scan every tile again and add the sum of all earlier tiles.
__global__ void scan_tiles(const int *in, int *out, const int *tile_offsets,
                           int n) {
  int items[ITEMS];
  load_items(in, n, blockIdx.x * TILE, items);
  scan_tile(items);
  const int offset = tile_offsets[blockIdx.x];
#pragma unroll
  for (int i = 0; i < ITEMS; ++i) {
    items[i] += offset;
  }
  store_items(out, n, blockIdx.x * TILE, items);
}

void launch(const int *d_in, int *d_out, int n, void *d_scratch,
            size_t /*scratch_bytes*/) {
  const int num_tiles = lab::ceil_div(n, TILE);
  int *tile_sums = static_cast<int *>(d_scratch);
  reduce_tiles<<<num_tiles, BLOCK>>>(d_in, tile_sums, n);
  scan_tile_sums<<<1, BLOCK>>>(tile_sums, num_tiles);
  scan_tiles<<<num_tiles, BLOCK>>>(d_in, d_out, tile_sums, n);
}

int main(int argc, char **argv) {
  return run_scan("Warp shuffles, reduce-then-scan", argc, argv, launch);
}
