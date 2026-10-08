# Sorting

| Step | Algorithm | Notes |
| --- | --- | --- |
| [00_counting_sort](00_counting_sort/00_counting_sort.cu) | Counting sort for small integer ranges | Histogram, CPU scan, atomic placement |
| [01_merge_sort](01_merge_sort/01_merge_sort.cu) | Merge sort | Bitonic sort of 1024-element runs in shared memory, then co-rank based parallel merges |
| [02_bitonic_sort](02_bitonic_sort/02_bitonic_sort.cu) | Bitonic sorting network | Data-independent compare-and-swap steps; shared memory for small strides, global passes for large ones |
| [03_radix_sort](03_radix_sort/03_radix_sort.cu) | LSD radix sort, 4-bit digits | Built from this repo's primitives: per-tile digit histograms, a scan for offsets, and a stable scatter ranked with `__match_any_sync` |
| [04_cub_radix_sort](04_cub_radix_sort/04_cub_radix_sort.cu) | `cub::DeviceRadixSort` | The baseline (Onesweep) |

Steps 02-04 sort the same random 32-bit keys through
[sort_harness.cuh](sort_harness.cuh) and report millions of keys per second.
