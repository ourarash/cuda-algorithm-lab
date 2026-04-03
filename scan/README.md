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
