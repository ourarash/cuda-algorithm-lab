/*
 * LSD Radix Sort, Built from Histogram and Scan
 *
 * Intention:
 * Radix sort is the fastest way to sort integer keys on a GPU, and it is
 * assembled from two primitives already in this repo: a histogram
 * (histogram/) and a prefix scan (scan/). It does O(n * passes) work and no
 * comparisons.
 *
 * Least-significant-digit (LSD) radix sort:
 * - Sort by the lowest 4 bits, then the next 4, ..., 8 passes for 32-bit
 *   keys. Each pass must be *stable* (equal digits keep their relative
 *   order), so the order established by earlier passes survives.
 *
 * One pass (digit = (key >> shift) & 15), with the input cut into tiles of
 * 1024 keys:
 * 1. tile_digit_counts: each tile counts how many of its keys have each of
 *    the 16 digits (a 16-bin histogram in shared memory) and stores them
 *    digit-major: counts[digit * num_tiles + tile].
 * 2. scan_counts: an exclusive scan over that array. Because it is
 *    digit-major, the result for (digit, tile) is the number of keys with a
 *    smaller digit anywhere, plus keys with the same digit in earlier tiles:
 *    exactly where this tile's keys with that digit start in the output.
 * 3. scatter: each tile writes every key to
 *      offset[digit][tile] + (number of earlier keys in this tile with the
 *                             same digit)
 *    The rank among equal digits is computed without sorting:
 *    - the tile is processed in rounds of 256 keys, in index order;
 *    - inside a warp, __match_any_sync(digit) returns the lanes holding the
 *      same digit, and __popc of the lanes below gives the rank in the warp;
 *    - per-warp digit counts in shared memory give each warp's offset within
 *      the round, and a running count per digit carries across rounds.
 *
 * Keys ping-pong between the output buffer and a scratch buffer; after the
 * eighth pass they are in the output. Production radix sorts (CUB, Onesweep)
 * use 8-bit digits, sort each tile in shared memory first so the global
 * writes are coalesced, and fuse the passes with a decoupled look-back scan.
 *
 * Requirements: compute capability 7.0+ (__match_any_sync).
 */
#include "../sort_harness.cuh"

constexpr int RADIX_BITS = 4;
constexpr int RADIX = 1 << RADIX_BITS;  // 16 digits
constexpr int PASSES = 32 / RADIX_BITS;
constexpr int BLOCK = 256;
constexpr int ROUNDS = 4;
constexpr int TILE = BLOCK * ROUNDS;  // 1024 keys per tile
constexpr int WARPS = BLOCK / 32;
constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

__device__ __forceinline__ int digit_of(unsigned int key, int shift) {
  return (key >> shift) & (RADIX - 1);
}

// 1. Per-tile digit histogram, stored digit-major.
__global__ void tile_digit_counts(const unsigned int *keys, int n, int shift,
                                  int *counts, int num_tiles) {
  __shared__ int local[RADIX];
  if (threadIdx.x < RADIX) {
    local[threadIdx.x] = 0;
  }
  __syncthreads();
  const int base = blockIdx.x * TILE;
  for (int r = 0; r < ROUNDS; ++r) {
    const int i = base + r * BLOCK + threadIdx.x;
    if (i < n) {
      atomicAdd(&local[digit_of(keys[i], shift)], 1);
    }
  }
  __syncthreads();
  if (threadIdx.x < RADIX) {
    counts[threadIdx.x * num_tiles + blockIdx.x] = local[threadIdx.x];
  }
}

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

// 2. Exclusive scan of the digit-major counts, in place, by one block.
__global__ void scan_counts(int *counts, int total) {
  constexpr int ITEMS = 4;
  int carry = 0;
  for (int base = 0; base < total; base += BLOCK * ITEMS) {
    const int first = base + threadIdx.x * ITEMS;
    int items[ITEMS];
    int thread_sum = 0;
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
      items[i] = first + i < total ? counts[first + i] : 0;
      thread_sum += items[i];
    }
    int chunk_total;
    int running = carry + block_exclusive_scan(thread_sum, chunk_total);
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
      if (first + i < total) {
        counts[first + i] = running;
      }
      running += items[i];
    }
    carry += chunk_total;
  }
}

// 3. Stable scatter of each tile to its digits' output ranges.
__global__ void scatter(const unsigned int *keys_in, unsigned int *keys_out,
                        int n, int shift, const int *offsets, int num_tiles) {
  __shared__ int tile_offset[RADIX];          // Where each digit starts
  __shared__ int warp_counts[WARPS][RADIX];   // Per round: counts, then prefixes
  __shared__ int round_total[RADIX];

  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const unsigned lanes_below = (1u << lane) - 1u;
  if (threadIdx.x < RADIX) {
    tile_offset[threadIdx.x] = offsets[threadIdx.x * num_tiles + blockIdx.x];
  }

  const int base = blockIdx.x * TILE;
  for (int r = 0; r < ROUNDS; ++r) {
    for (int k = threadIdx.x; k < WARPS * RADIX; k += BLOCK) {
      warp_counts[k / RADIX][k % RADIX] = 0;
    }
    __syncthreads();

    const int i = base + r * BLOCK + threadIdx.x;
    const bool valid = i < n;
    const unsigned int key = valid ? keys_in[i] : 0;
    // Invalid lanes get digit -1 so they never match a real digit.
    const int digit = valid ? digit_of(key, shift) : -1;
    const unsigned peers = __match_any_sync(FULL_MASK, digit);
    const int rank_in_warp = __popc(peers & lanes_below);
    if (valid && rank_in_warp == 0) {  // Lowest lane of each digit group
      warp_counts[warp][digit] = __popc(peers);
    }
    __syncthreads();

    // One thread per digit: exclusive prefix over the warps, in warp order.
    if (threadIdx.x < RADIX) {
      int running = 0;
      for (int w = 0; w < WARPS; ++w) {
        const int c = warp_counts[w][threadIdx.x];
        warp_counts[w][threadIdx.x] = running;
        running += c;
      }
      round_total[threadIdx.x] = running;
    }
    __syncthreads();

    if (valid) {
      keys_out[tile_offset[digit] + warp_counts[warp][digit] + rank_in_warp] = key;
    }
    __syncthreads();

    // Later rounds of this tile come after this round's keys.
    if (threadIdx.x < RADIX) {
      tile_offset[threadIdx.x] += round_total[threadIdx.x];
    }
    __syncthreads();
  }
}

void launch(const unsigned int *d_in, unsigned int *d_out, int n,
            void *d_scratch, size_t scratch_bytes) {
  const int num_tiles = lab::ceil_div(n, TILE);
  unsigned int *tmp = static_cast<unsigned int *>(d_scratch);
  int *counts = reinterpret_cast<int *>(tmp + n);
  if (static_cast<size_t>(n) * sizeof(unsigned int) +
          static_cast<size_t>(RADIX) * num_tiles * sizeof(int) > scratch_bytes) {
    std::fprintf(stderr, "radix sort needs more scratch\n");
    std::exit(EXIT_FAILURE);
  }

  const unsigned int *src = d_in;
  for (int pass = 0; pass < PASSES; ++pass) {
    // Alternate buffers so that the last pass writes into d_out.
    unsigned int *dst = (PASSES - 1 - pass) % 2 == 0 ? d_out : tmp;
    const int shift = pass * RADIX_BITS;
    tile_digit_counts<<<num_tiles, BLOCK>>>(src, n, shift, counts, num_tiles);
    scan_counts<<<1, BLOCK>>>(counts, RADIX * num_tiles);
    scatter<<<num_tiles, BLOCK>>>(src, dst, n, shift, counts, num_tiles);
    src = dst;
  }
}

int main(int argc, char **argv) {
  if (lab::compute_capability() < 70) {
    return lab::skip("__match_any_sync needs compute capability 7.0 or newer");
  }
  return run_sort("LSD radix sort (4-bit digits)", argc, argv, launch);
}
