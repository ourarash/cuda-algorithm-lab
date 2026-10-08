# Memory and Concurrency

How data gets to the GPU, and how to keep the GPU busy while it does.

| Example | What it shows |
| --- | --- |
| [00_unified_memory_vector_add](00_unified_memory_vector_add/00_unified_memory_vector_add.cu) | Managed memory: one pointer for CPU and GPU |
| [01_pinned_vs_pageable](01_pinned_vs_pageable/01_pinned_vs_pageable.cu) | Host-to-device bandwidth from `malloc` memory vs. `cudaMallocHost` (pinned) memory |
| [02_streams_overlap](02_streams_overlap/02_streams_overlap.cu) | Splitting work into chunks over several streams so copies in, kernels, and copies out overlap |
| [03_cuda_graphs](03_cuda_graphs/03_cuda_graphs.cu) | Capturing a sequence of small kernels into a CUDA graph to cut launch overhead |
| [04_unified_memory_prefetch](04_unified_memory_prefetch/04_unified_memory_prefetch.cu) | On-demand page faults vs. `cudaMemPrefetchAsync` for managed memory (skipped where page faulting is unsupported, such as Windows) |

Nsight Systems (`nsys profile ./02_streams_overlap`) shows the overlap on a
timeline; it is the right tool for this kind of question, where Nsight
Compute looks inside a single kernel.
