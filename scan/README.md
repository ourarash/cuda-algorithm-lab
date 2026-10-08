# CUDA Scan Evolution

This folder now mirrors the progressive layout used in `matmul/` and
`reduction/`. Each step keeps the same core prefix-sum problem but improves a
different part of the implementation.

## Files Overview

### 0. `00_kogge_stone/00_scan_kogge_stone.cu`

**Kogge-Stone Inclusive Scan**

- The simplest single-block GPU scan in this folder.
- Uses recursive doubling directly in shared memory.
- Easy to understand, but performs `O(n log n)` work and needs two barriers per
  stage.

### 1. `01_hillis_steele_double_buffer/01_scan_hillis_steele_double_buffer.cu`

**Hillis-Steele with Double Buffering**

- Still `O(n log n)` work, but reads from one shared-memory buffer and writes to
  another.
- This makes the data flow easier to reason about because each stage reads a
  stable snapshot from the previous stage.

### 2. `02_brent_kung/02_scan_brent_kung.cu`

**Brent-Kung Inclusive Scan**

- This is the classic scan algorithm that was missing from the original folder.
- It reduces the amount of work compared with Kogge-Stone/Hillis-Steele by
  using a reduce phase followed by a distribute phase.

### 3. `03_blelloch/03_scan_blelloch.cu`

**Blelloch Exclusive Scan**

- A work-efficient tree scan with an up-sweep and down-sweep.
- This is the standard exclusive scan formulation used in many CUDA teaching
  materials.

### 4. `04_blelloch_bank_conflict_free/04_scan_blelloch_bank_conflict_free.cu`

**Blelloch with Bank-Conflict Padding**

- Same algorithm as the previous step, but with padded shared-memory indices to
  reduce serialization from shared-memory bank conflicts.

### 5. `05_multiblock_cpu_fixup/05_scan_multiblock_cpu_fixup.cu`

**Large-Array Scan with CPU Block Fixup**

- Extends scan beyond a single block by scanning each block on the GPU, then
  scanning block sums on the CPU and applying the block offsets on the host.

### 6. `06_multiblock_gpu_fixup/06_scan_multiblock_gpu_fixup.cu`

**Large-Array Scan with Recursive GPU Fixup**

- Finishes the multi-block story entirely on the GPU.
- Recursively scans the array of block sums, then adds scanned block offsets
  back into each block output.

### 7. `07_warp_shuffle_reduce_then_scan/07_scan_warp_shuffle.cu`

**Warp Shuffles, Reduce-Then-Scan**

- Integer scan with the modern building blocks: each thread scans 4 elements
  in registers (one 16-byte load), warps scan with `__shfl_up_sync`, and
  warp totals are combined through shared memory.
- Any array size takes exactly three launches: tile totals, a single-block
  scan of the totals, and a final scan of each tile with its offset. The input
  is read twice.

### 8. `08_decoupled_lookback/08_scan_decoupled_lookback.cu`

**Single-Pass Scan with Decoupled Look-Back**

- The algorithm inside CUB (Merrill and Garland): each tile publishes its
  total, then looks back at its predecessors' published values to find its
  prefix, so the input is read once.
- Tiles are numbered by an atomic counter in the order blocks start, which
  guarantees that the spin-wait on an earlier tile cannot deadlock.

### 9. `09_cub/09_scan_cub.cu`

**CUB DeviceScan (the baseline)**

- `cub::DeviceScan::InclusiveSum`, the tuned version of step 08.

Steps 07-09 share [scan_harness.cuh](scan_harness.cuh): exact validation on
integers and GB/s against the minimum traffic (read once, write once).
