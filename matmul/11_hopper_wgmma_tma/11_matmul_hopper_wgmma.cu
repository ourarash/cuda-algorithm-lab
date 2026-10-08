/*
 * Hopper Matrix Multiplication with TMA and WGMMA (sm_90a)
 *
 * Intention:
 * Hopper (H100, compute capability 9.0) adds two features that change how a
 * fast GEMM is written:
 * - TMA (Tensor Memory Accelerator): one thread describes a whole 2D tile
 *   and a dedicated copy engine moves it from global to shared memory. No
 *   per-thread address math, no per-thread copies, and out-of-range parts of
 *   a tile are filled with zeros by the hardware.
 * - WGMMA (warpgroup matrix multiply-accumulate): four warps (a "warpgroup",
 *   128 threads) issue one asynchronous Tensor Core instruction that reads A
 *   and B directly from shared memory and accumulates a 64 x N tile in
 *   registers. There is no ldmatrix step.
 *
 * High-Level Algorithm:
 * - Block tile 128 x 128, BK = 64. Two warpgroups (256 threads); warpgroup w
 *   computes rows 64w .. 64w + 63 of the block tile, all 128 columns.
 * - Per K tile:
 *     thread 0: tell the mbarrier to expect 32 KB, then issue two TMA loads
 *               (A tile 128 x 64 and B tile 128 x 64); TMA signals the
 *               mbarrier as bytes arrive.
 *     all:      wait on the mbarrier until the tile has landed.
 *     each warpgroup: 4 x wgmma.m64n128k16 (one per 16-wide K step), commit,
 *               wait for completion; __syncthreads() before the next load.
 * - Epilogue: each thread holds 64 FP32 accumulators in the documented
 *   m64nNk16 layout and applies C = alpha * acc + beta * C.
 *
 * Layout requirements:
 * - WGMMA reads both operands "K-major" here: each row of the A tile and of
 *   the B tile holds consecutive K values. B is therefore passed to this
 *   kernel transposed, as an N x K row-major array (the harness does this
 *   when b_k_major is set; cuBLAS and the reference still use B as K x N).
 * - TMA writes the tiles with 128-byte swizzling (each 128-byte row has its
 *   16-byte chunks XOR-permuted by row % 8, the same idea as stage 10), and
 *   the WGMMA shared-memory descriptors declare the same swizzle so the
 *   Tensor Cores read the data back in the right order. A 64-half row is
 *   exactly 128 bytes, and swizzled tiles must start on a 1024-byte boundary.
 * - K must be a multiple of 8 (TMA needs 16-byte row strides). M and N can
 *   be anything: TMA zero-fills tiles at the edges and the epilogue checks
 *   bounds.
 *
 * This is the simplest correct TMA + WGMMA kernel: one shared-memory stage
 * and no overlap between loading and computing. Production Hopper GEMMs add
 * a multi-stage pipeline with producer and consumer warpgroups, thread block
 * clusters with TMA multicast, and larger tiles.
 *
 * Validation status: compiled for sm_90a in CI, which assembles the PTX, but
 * not yet run on Hopper hardware. The harness validates the result against
 * the CPU on every run and exits non-zero on any mismatch.
 *
 * Requirements: compute capability exactly 9.0 (sm_90a code runs only on
 * Hopper), CUDA 12.0+, and the driver API (libcuda) for cuTensorMapEncodeTiled.
 */
#include <cuda.h>

#include <cstdint>

#include "../gemm_harness.cuh"

#define CU_CHECK(call)                                                     \
  do {                                                                     \
    CUresult res_ = (call);                                                \
    if (res_ != CUDA_SUCCESS) {                                            \
      const char *msg_ = nullptr;                                          \
      cuGetErrorString(res_, &msg_);                                       \
      std::fprintf(stderr, "CUDA driver error %s at %s:%d\n",              \
                   msg_ ? msg_ : "?", __FILE__, __LINE__);                 \
      std::exit(EXIT_FAILURE);                                             \
    }                                                                      \
  } while (0)

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 64;  // 64 halves = 128 bytes = one swizzle row
constexpr int WARPGROUPS = 2;
constexpr int NUM_THREADS = WARPGROUPS * 128;
constexpr int WG_M = BM / WARPGROUPS;  // 64 rows per warpgroup (wgmma M)
constexpr int WGMMA_K = 16;
constexpr int A_TILE_BYTES = BM * BK * 2;
constexpr int B_TILE_BYTES = BN * BK * 2;
constexpr int SMEM_ALIGN = 1024;
constexpr int SMEM_BYTES = A_TILE_BYTES + B_TILE_BYTES + SMEM_ALIGN;  // + slack

static_assert(BK * 2 == 128, "128-byte swizzle needs 128-byte tile rows");
static_assert(WG_M == 64, "wgmma.m64nNk16 computes 64 rows per warpgroup");
static_assert(A_TILE_BYTES % SMEM_ALIGN == 0, "B tile must stay 1024-aligned");

__device__ __forceinline__ uint32_t smem_addr(const void *p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

#if defined(__CUDA_ARCH_FEAT_SM90_ALL)

// ---- mbarrier: a shared-memory barrier that TMA can signal ----
__device__ __forceinline__ void mbarrier_init(uint64_t *bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_addr(bar)),
               "r"(count));
  // Make the initialized barrier visible to the TMA unit.
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

// Arrive once and announce how many bytes TMA will deliver this phase.
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t *bar,
                                                          uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(
                   smem_addr(bar)),
               "r"(bytes)
               : "memory");
}

// Spin until the phase with the given parity has completed.
__device__ __forceinline__ void mbarrier_wait(uint64_t *bar, uint32_t parity) {
  uint32_t done = 0;
  do {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        "selp.u32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(done)
        : "r"(smem_addr(bar)), "r"(parity)
        : "memory");
  } while (!done);
}

// ---- TMA: copy one box of a 2D tensor into shared memory ----
// Coordinates are (innermost, outer) = (k, row).
__device__ __forceinline__ void tma_load_2d(void *dst, const CUtensorMap *map,
                                            int k, int row, uint64_t *bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];\n" ::"r"(smem_addr(dst)),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(k), "r"(row),
      "r"(smem_addr(bar))
      : "memory");
}

// ---- WGMMA ----
// Shared-memory matrix descriptor for a K-major tile with 128-byte swizzling:
//   bits  0-13: start address >> 4
//   bits 16-29: leading-dimension byte offset >> 4 (unused for this layout)
//   bits 32-45: stride byte offset >> 4: 1024 bytes between 8-row groups
//   bits 62-63: swizzle mode, 1 = 128-byte swizzle
// Advancing along K inside the 128-byte row just moves the start address
// (32 bytes per 16 halves); the hardware applies the swizzle to the final
// address, matching how TMA placed the data.
__device__ __forceinline__ uint64_t make_smem_desc(const half *p) {
  const uint64_t addr = smem_addr(p);
  uint64_t desc = 0;
  desc |= (addr & 0x3FFFF) >> 4;
  desc |= static_cast<uint64_t>(16 >> 4) << 16;
  desc |= static_cast<uint64_t>(1024 >> 4) << 32;
  desc |= static_cast<uint64_t>(1) << 62;
  return desc;
}

__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
template <int N>
__device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

// Keeps the compiler from moving accumulator reads or writes across the
// asynchronous wgmma, which updates these registers behind its back.
__device__ __forceinline__ void fence_accumulators(float (&d)[64]) {
#pragma unroll
  for (int i = 0; i < 64; ++i) {
    asm volatile("" : "+f"(d[i])::"memory");
  }
}

// d (64 x 128, FP32) += A (64 x 16, FP16) * B (16 x 128, FP16), both operands
// read from shared memory through descriptors, both K-major (no transpose).
__device__ __forceinline__ void wgmma_m64n128k16(float (&d)[64], uint64_t desc_a,
                                                 uint64_t desc_b) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
      "%64, %65, p, 1, 1, 0, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(desc_a), "l"(desc_b), "r"(1));
}

#endif  // __CUDA_ARCH_FEAT_SM90_ALL

/**
 * 11. Hopper: TMA + WGMMA
 */
__global__ void __launch_bounds__(NUM_THREADS)
    hgemm_wgmma_tma(int M, int N, int K, float alpha, float beta, float *C,
                    const __grid_constant__ CUtensorMap tmap_a,
                    const __grid_constant__ CUtensorMap tmap_b) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  __shared__ uint64_t bar;

  uint8_t *smem = reinterpret_cast<uint8_t *>(
      (reinterpret_cast<uintptr_t>(smem_raw) + SMEM_ALIGN - 1) &
      ~static_cast<uintptr_t>(SMEM_ALIGN - 1));
  half *sA = reinterpret_cast<half *>(smem);                 // BM x BK, swizzled
  half *sB = reinterpret_cast<half *>(smem + A_TILE_BYTES);  // BN x BK, swizzled

  const int wg = threadIdx.x / 128;
  const int blockRow = blockIdx.y * BM;
  const int blockCol = blockIdx.x * BN;

  if (threadIdx.x == 0) {
    mbarrier_init(&bar, 1);
  }
  __syncthreads();

  float d[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) {
    d[i] = 0.0f;
  }

  const int numTiles = lab::ceil_div(K, BK);
  for (int t = 0; t < numTiles; ++t) {
    if (threadIdx.x == 0) {
      // TMA always writes full boxes (zero-filling past the matrix edge), so
      // the byte count is the same for every tile.
      mbarrier_arrive_expect_tx(&bar, A_TILE_BYTES + B_TILE_BYTES);
      tma_load_2d(sA, &tmap_a, t * BK, blockRow, &bar);
      tma_load_2d(sB, &tmap_b, t * BK, blockCol, &bar);
    }
    mbarrier_wait(&bar, t & 1);

    // This warpgroup's 64 rows of A start 64 * 128 bytes into the tile.
    const half *a_rows = sA + wg * WG_M * BK;
    fence_accumulators(d);
    wgmma_fence();
#pragma unroll
    for (int kk = 0; kk < BK / WGMMA_K; ++kk) {
      wgmma_m64n128k16(d, make_smem_desc(a_rows + kk * WGMMA_K),
                       make_smem_desc(sB + kk * WGMMA_K));
    }
    wgmma_commit();
    wgmma_wait<0>();
    fence_accumulators(d);

    // Both warpgroups must be done reading the tiles before thread 0 starts
    // the next TMA into the same buffers.
    __syncthreads();
  }

  // ---- Epilogue ----
  // m64nNk16 accumulator layout: warp w of the warpgroup owns rows
  // 16w .. 16w + 15. For each 8-column block i, d[4i + 0..1] are in row
  // lane / 4 and d[4i + 2..3] in row lane / 4 + 8, at columns
  // 8i + 2 * (lane % 4) + {0, 1}.
  const int warp = (threadIdx.x % 128) / 32;
  const int lane = threadIdx.x % 32;
#pragma unroll
  for (int j = 0; j < 64; ++j) {
    const int row = blockRow + wg * WG_M + warp * 16 + lane / 4 + ((j % 4) / 2) * 8;
    const int col = blockCol + (j / 4) * 8 + (lane % 4) * 2 + (j % 2);
    if (row < M && col < N) {
      float &c = C[static_cast<size_t>(row) * N + col];
      c = alpha * d[j] + beta * c;
    }
  }
#elif defined(__CUDA_ARCH__)
  __trap();  // Built only for sm_90a; the host never launches it elsewhere.
#endif
}

// Describes a rows x k row-major FP16 matrix to TMA, with 128 x 64 boxes and
// 128-byte swizzling to match the kernel's shared-memory descriptors.
static CUtensorMap make_tensor_map(const half *ptr, int rows, int k) {
  CUtensorMap map;
  const cuuint64_t global_dim[2] = {static_cast<cuuint64_t>(k),
                                    static_cast<cuuint64_t>(rows)};
  const cuuint64_t global_stride[1] = {static_cast<cuuint64_t>(k) * sizeof(half)};
  const cuuint32_t box[2] = {BK, 128};
  const cuuint32_t element_stride[2] = {1, 1};
  CU_CHECK(cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, const_cast<half *>(ptr),
      global_dim, global_stride, box, element_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return map;
}

// B arrives as an N x K row-major array (see the header comment).
void launch_hgemm_wgmma_tma(int M, int N, int K, float alpha, const half *A,
                            const half *B, float beta, float *C) {
  // Encoding tensor maps is host work; cache them so timed launches measure
  // only the kernel.
  static const half *cached_a = nullptr;
  static const half *cached_b = nullptr;
  static int cached_m = 0, cached_n = 0, cached_k = 0;
  static CUtensorMap map_a, map_b;
  if (A != cached_a || B != cached_b || M != cached_m || N != cached_n ||
      K != cached_k) {
    map_a = make_tensor_map(A, M, K);
    map_b = make_tensor_map(B, N, K);
    cached_a = A;
    cached_b = B;
    cached_m = M;
    cached_n = N;
    cached_k = K;
  }
  dim3 grid(lab::ceil_div(N, BN), lab::ceil_div(M, BM));
  hgemm_wgmma_tma<<<grid, NUM_THREADS, SMEM_BYTES>>>(M, N, K, alpha, beta, C,
                                                     map_a, map_b);
}

int main(int argc, char **argv) {
  GemmRequirements req;
  req.k_multiple = 8;
  req.exact_compute_capability = 90;
  req.b_k_major = true;
  return run_gemm<half>("Hopper TMA + WGMMA", argc, argv, {1024, 1024, 1024},
                        {200, 136, 72}, launch_hgemm_wgmma_tma, req);
}
