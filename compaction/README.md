# Stream Compaction

Copy the elements that satisfy a predicate into a dense array and report how
many there were (also called filtering or select). It is a building block for
many GPU algorithms: removing finished work items, culling, building sparse
structures, and radix sort's scatter step.

All steps keep the ints below 300 from uniform random input in [0, 1000)
and share [compaction_harness.cuh](compaction_harness.cuh). Stable results
(input order preserved) are checked exactly; the unstable one is checked as a
multiset.

| Step | What changes | Order |
| --- | --- | --- |
| [00_scan_based](00_scan_based/00_compaction_scan_based.cu) | An element's output position is the exclusive scan of the keep flags before it: count per tile, scan the counts, scatter (the structure of `scan/07`) | Stable |
| [01_warp_aggregated_atomics](01_warp_aggregated_atomics/01_compaction_warp_aggregated.cu) | `__ballot_sync` finds the selected lanes; one `atomicAdd` per warp reserves their slots; `__popc` ranks lanes within the warp | Unstable |
| [02_cub](02_cub/02_compaction_cub.cu) | `cub::DeviceSelect::If` (single pass with decoupled look-back), the baseline | Stable |
