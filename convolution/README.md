# 2D Convolution

A 7 x 7 filter over a float image with zero padding, the core of image
filtering and of convolutional layers. All steps share
[convolution_harness.cuh](convolution_harness.cuh) (CPU reference, GFLOP/s and
GB/s).

| Step | What changes | Nsight Compute evidence |
| --- | --- | --- |
| [00_naive](00_naive/00_convolution_naive.cu) | One thread per pixel; image and filter read from global memory | Each input value is requested up to 49 times (L1/L2 hit rates show how much the caches absorb) |
| [01_constant_memory](01_constant_memory/01_convolution_constant_memory.cu) | Filter in `__constant__` memory: one broadcast per warp; loops unrolled | Global load requests drop by the filter reads |
| [02_shared_memory_tiles](02_shared_memory_tiles/02_convolution_shared_tiles.cu) | Each block loads its 32 x 8 tile plus a 3-pixel halo into shared memory once | Global loads drop to about 2 per input value; the inner loop has no bounds checks |

See [stencil/](../stencil/) for the 3D version of the same idea, where the
streaming dimension adds a register-tiling trick.
