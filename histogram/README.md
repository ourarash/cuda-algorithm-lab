# Histogram

Counting how often each byte value occurs: 256 bins, many threads, and the
classic lesson in atomic contention. Every step runs two inputs, because the
best technique depends on the data:

- **uniform** random bytes, where updates rarely collide;
- **skewed** image-like data with long runs and a few dominant values, where
  many threads hit the same bins at once.

All steps share [histogram_harness.cuh](histogram_harness.cuh), which checks
the counts exactly and reports GB/s of input.

| Step | What changes | Nsight Compute evidence |
| --- | --- | --- |
| [00_global_atomics](00_global_atomics/00_histogram_global_atomics.cu) | One global `atomicAdd` per element | Throughput collapses on the skewed input: atomics to the same address serialize in L2 |
| [01_shared_privatization](01_shared_privatization/01_histogram_shared_privatization.cu) | A private histogram per block in shared memory, merged once at the end | Global atomics drop from one per element to 256 per block |
| [02_aggregation](02_aggregation/02_histogram_aggregation.cu) | 16-byte loads; runs of equal values counted in a register and added with one atomic | Shared-memory atomics drop sharply on the skewed input |
| [03_cub](03_cub/03_histogram_cub.cu) | `cub::DeviceHistogram::HistogramEven`, the baseline | |

Privatization (02's private copies) and aggregation (combining updates before
they hit shared state) are general patterns for any reduction into a small
output. See *Programming Massively Parallel Processors*, chapter 9.
