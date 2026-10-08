/*
 * Single-Pass Scan with Decoupled Look-Back
 *
 * Intention:
 * Step 07 reads the input twice: once to compute tile totals, once to scan.
 * Merrill and Garland's "Single-pass Parallel Prefix Scan with Decoupled
 * Look-back" (the algorithm inside CUB) scans in one pass, so the input is
 * read once and the output written once, the minimum possible traffic.
 *
 * High-Level Algorithm:
 * - Each block takes the next tile number from an atomic counter, scans its
 *   tile exactly as in step 07, and knows its own tile total (aggregate).
 * - Every tile has a status word in global memory holding a flag and a value:
 *     NOT_READY  nothing published yet
 *     AGGREGATE  value = this tile's own total
 *     PREFIX     value = total of this tile and every tile before it
 * - Tile 0 publishes PREFIX immediately. Every other tile first publishes
 *   AGGREGATE (so successors can use it right away), then looks back:
 *   starting at the previous tile it adds up AGGREGATE values until it meets
 *   a PREFIX, which already covers everything further back. It then
 *   publishes its own PREFIX and adds the exclusive prefix to its items.
 * - The look-back usually stops after one or two tiles, because the tile
 *   just before it has typically already published its PREFIX.
 *
 * Why the atomic tile counter instead of blockIdx.x: a block spins until
 * earlier tiles publish. The GPU may schedule blocks in any order, so if
 * tiles followed blockIdx.x, a block could wait on a tile whose block has not
 * started and never will start, because all SMs are busy with waiting blocks.
 * Taking tile numbers in the order blocks actually start guarantees every
 * earlier tile belongs to a block that is already running.
 *
 * Status words pack the flag (high 32 bits) and value (low 32 bits) into one
 * 64-bit word, written with a single atomic store and read with a single
 * volatile load, so a reader never sees a flag without its value.
 *
 * This version lets one thread do the look-back; CUB uses a whole warp to
 * inspect 32 predecessors at once.
 */
#include "../scan_harness.cuh"

constexpr int BLOCK = 256;
constexpr int ITEMS = 4;
constexpr int TILE = BLOCK * ITEMS;
constexpr int WARPS = BLOCK / 32;
constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

constexpr unsigned long long NOT_READY = 0;
constexpr unsigned long long AGGREGATE = 1;
constexpr unsigned long long PREFIX = 2;

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

__device__ __forceinline__ unsigned long long pack(unsigned long long flag,
                                                   int value) {
  return (flag << 32) | static_cast<unsigned int>(value);
}

__global__ void scan_decoupled_lookback(const int *in, int *out, int n,
                                        unsigned long long *status,
                                        unsigned int *tile_counter) {
  __shared__ int tile_s;
  __shared__ int exclusive_prefix_s;

  if (threadIdx.x == 0) {
    tile_s = static_cast<int>(atomicAdd(tile_counter, 1u));
  }
  __syncthreads();
  const int tile = tile_s;
  const int base = tile * TILE;

  // ---- Scan this tile (as in step 07) ----
  int items[ITEMS];
  const int first = base + threadIdx.x * ITEMS;
  if (first + ITEMS <= n) {
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
  int thread_sum = 0;
#pragma unroll
  for (int i = 0; i < ITEMS; ++i) {
    thread_sum += items[i];
    items[i] = thread_sum;
  }
  int aggregate;
  const int thread_prefix = block_exclusive_scan(thread_sum, aggregate);

  // ---- Publish and look back (one thread) ----
  if (threadIdx.x == 0) {
    int exclusive = 0;
    if (tile == 0) {
      atomicExch(&status[0], pack(PREFIX, aggregate));
    } else {
      atomicExch(&status[tile], pack(AGGREGATE, aggregate));
      int pred = tile - 1;
      while (true) {
        const unsigned long long s =
            *reinterpret_cast<volatile unsigned long long *>(&status[pred]);
        const unsigned long long flag = s >> 32;
        if (flag == NOT_READY) {
          continue;  // The predecessor is running; wait for it to publish.
        }
        exclusive += static_cast<int>(static_cast<unsigned int>(s));
        if (flag == PREFIX) {
          break;
        }
        --pred;  // Only an aggregate: keep walking back.
      }
      atomicExch(&status[tile], pack(PREFIX, exclusive + aggregate));
    }
    exclusive_prefix_s = exclusive;
  }
  __syncthreads();

  // ---- Write the tile with everything before it added ----
  const int offset = exclusive_prefix_s + thread_prefix;
#pragma unroll
  for (int i = 0; i < ITEMS; ++i) {
    items[i] += offset;
  }
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

void launch(const int *d_in, int *d_out, int n, void *d_scratch,
            size_t /*scratch_bytes*/) {
  const int num_tiles = lab::ceil_div(n, TILE);
  auto *status = static_cast<unsigned long long *>(d_scratch);
  auto *tile_counter = reinterpret_cast<unsigned int *>(status + num_tiles);
  // Every launch needs fresh status words and a counter starting at 0.
  CUDA_CHECK(cudaMemsetAsync(d_scratch, 0,
                             num_tiles * sizeof(unsigned long long) + sizeof(unsigned int)));
  scan_decoupled_lookback<<<num_tiles, BLOCK>>>(d_in, d_out, n, status,
                                                tile_counter);
}

int main(int argc, char **argv) {
  return run_scan("Single pass, decoupled look-back", argc, argv, launch);
}
