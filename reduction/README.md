# Parallel Reduction Ladder

Thirteen steps that sum an array of floats on the GPU, from a kernel that
works directly in global memory to NVIDIA's CUB library. Steps 01-07 follow
Mark Harris's classic
[Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
(updated for GPUs since Volta, where warps no longer run in lockstep); steps
08-11 use hardware features added since, and step 12 is the library baseline.

Reduction does one add per 4 bytes read, so it is limited by memory bandwidth,
and the number that matters is GB/s as a percentage of the GPU's peak.

Every step shares one host-side driver,
[reduction_harness.cuh](reduction_harness.cuh). Each step computes the whole
sum on the GPU (repeating passes over the partial sums where needed), and the
harness checks it twice:

- **Exact check:** about a million small integers. Every partial sum is an
  integer below 2^24, which a float represents exactly, so the result must
  match exactly in any order of addition. A single lost or double-counted
  element fails.
- **Accuracy check:** 16 million random floats against a double-precision sum.

Pass `--quick` for a small size or `--n` to choose one.

## The ladder

| Step | What changes | Nsight Compute evidence |
| --- | --- | --- |
| [00_naive](00_naive/00_reduction_naive.cu) | Tree reduction in place in global memory, growing stride | All traffic goes to DRAM; very low `dram__throughput.avg.pct_of_peak_sustained_elapsed` despite many bytes moved |
| [01_interleaved_divergent](01_interleaved_divergent/01_reduction_interleaved_divergent.cu) | Tree in shared memory; `tid % (2s) == 0` picks scattered threads | Low active threads per instruction (`smsp__thread_inst_executed_per_inst_executed.ratio`): warp divergence |
| [02_interleaved_bank_conflicts](02_interleaved_bank_conflicts/02_reduction_interleaved_bank_conflicts.cu) | Consecutive threads work at stride `2s`: no divergence | Shared-memory bank conflicts (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`) |
| [03_sequential_addressing](03_sequential_addressing/03_reduction_sequential_addressing.cu) | Halving stride: active threads and addresses are contiguous | Bank conflicts drop to zero |
| [04_first_add_during_load](04_first_add_during_load/04_reduction_first_add_during_load.cu) | Each thread adds two elements while loading; half the blocks | DRAM throughput rises |
| [05_unroll_last_warp](05_unroll_last_warp/05_reduction_unroll_last_warp.cu) | The last six levels run in one warp with `__syncwarp()` instead of `__syncthreads()` | Fewer instructions (`smsp__inst_executed.sum`) and barrier stalls |
| [06_complete_unroll](06_complete_unroll/06_reduction_complete_unroll.cu) | Block size as a template parameter; the whole tree unrolled | Fewer instructions again |
| [07_grid_stride](07_grid_stride/07_reduction_grid_stride.cu) | A grid sized to the GPU; each thread sums many elements in a register | Far fewer blocks, so the tree's cost is paid far less often |
| [08_warp_shuffle](08_warp_shuffle/08_reduction_warp_shuffle.cu) | Warps sum with `__shfl_down_sync`; shared memory only between warps | Shared-memory instructions (`smsp__sass_inst_executed_op_shared_st.sum`) nearly vanish |
| [09_cooperative_groups](09_cooperative_groups/09_reduction_cooperative_groups.cu) | The same with `cg::reduce` over a 32-thread tile | Same as 08: the API, not the algorithm, changed |
| [10_single_pass](10_single_pass/10_reduction_single_pass.cu) | The last block to finish adds the block sums: one launch instead of two | One kernel instead of two in the Nsight Systems timeline |
| [11_vectorized](11_vectorized/11_reduction_vectorized.cu) | `float4` loads | Global load instructions (`smsp__sass_inst_executed_op_global_ld.sum`) drop about 4x |
| [12_cub](12_cub/12_reduction_cub.cu) | `cub::DeviceReduce::Sum`, the library baseline | The bar to beat |

## Notes

**05, the warp-synchronous trick and Volta.** Harris's original "unroll the
last warp" relied on the 32 threads of a warp running in lockstep and used a
`volatile` pointer with no synchronization. Since Volta, threads of a warp are
scheduled independently, so that code is no longer correct. Steps 05-07
separate each read and write with `__syncwarp()`, and step 08 removes the need
with shuffles.

**10, why not one `atomicAdd` per block?** It is also a single pass, but float
addition is not associative and the order of atomics changes between runs, so
the result would not be reproducible bit for bit. The last-block pattern adds
the partial sums in a fixed order.

## Results

Measure on your GPU with `make bench` and paste the table here, naming the GPU.
No measurements are committed yet: CI runners have no GPU.

## Visualizations

- [00_naive/naive_reduction_visualization_tree.html](00_naive/naive_reduction_visualization_tree.html): the growing-stride tree, warp divergence, and the alternative mappings
- [03_sequential_addressing/shared_reduction_visualization.html](03_sequential_addressing/shared_reduction_visualization.html): the shared-memory tree with sequential addressing
