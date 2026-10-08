# Matrix Transpose Ladder

Five steps that show the two memory-system lessons a transpose teaches:
coalescing global memory accesses, and avoiding shared-memory bank conflicts.
They follow NVIDIA's
[An Efficient Matrix Transpose in CUDA C/C++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/),
with one more step that replaces padding by swizzling.

A transpose reads and writes every element once, so it is bandwidth-bound.
Step 00 is a plain copy with the same tiling: the fastest a transpose could
possibly be on your GPU. All steps share
[transpose_harness.cuh](transpose_harness.cuh), which validates the result
exactly and reports GB/s with % of peak bandwidth. They all use 32 x 32 tiles
handled by 32 x 8 threads (4 elements per thread).

## The ladder

| Step | What changes | Nsight Compute evidence |
| --- | --- | --- |
| [00_copy](00_copy/00_transpose_copy.cu) | Copy without transposing: the speed limit | DRAM throughput (`dram__throughput.avg.pct_of_peak_sustained_elapsed`) near peak |
| [01_naive](01_naive/01_transpose_naive.cu) | `out[x][y] = in[y][x]`: reads coalesced, writes strided | Store sectors per request (`l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum` / `l1tex__t_requests_pipe_lsu_mem_global_op_st.sum`) far above 4 |
| [02_shared_memory](02_shared_memory/02_transpose_shared_memory.cu) | Transpose inside a shared-memory tile; global writes become coalesced | Store sectors per request drops to about 4, but 32-way bank conflicts appear (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`) |
| [03_padded](03_padded/03_transpose_padded.cu) | Tile declared `[32][33]`: a column now spans all 32 banks | Bank conflicts drop to zero |
| [04_swizzled](04_swizzled/04_transpose_swizzled.cu) | Element (r, c) stored at column `c ^ r` instead of padding | Bank conflicts stay at zero, with no wasted shared memory |

## Results

Measure on your GPU with `make bench` and paste the table here, naming the GPU.
No measurements are committed yet: CI runners have no GPU.

## Visualization

[03_padded/03_transpose_bank_conflicts_visualization.html](03_padded/03_transpose_bank_conflicts_visualization.html)
shows how a column read maps onto shared-memory banks with and without padding.
