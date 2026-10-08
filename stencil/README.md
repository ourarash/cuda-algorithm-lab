# 3D Stencil

One step of a 7-point heat-equation (Jacobi) update on an N^3 grid: each
interior point becomes a weighted sum of itself and its six neighbours. Both
steps share [stencil_harness.cuh](stencil_harness.cuh).

| Step | What changes | Nsight Compute evidence |
| --- | --- | --- |
| [00_naive](00_naive/00_stencil_naive.cu) | One thread per point, 7 global loads each | z-neighbours are a whole plane apart and rarely cached: DRAM traffic well above 2x the grid size |
| [01_shared_memory_z_streaming](01_shared_memory_z_streaming/01_stencil_z_streaming.cu) | A block marches along z: the current x-y plane (with halo) in shared memory, the planes above and below in registers | DRAM traffic close to one read and one write of the grid |

See *Programming Massively Parallel Processors*, chapter 8, for thread
coarsening and register tiling in stencils.
