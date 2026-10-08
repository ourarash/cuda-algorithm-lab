# Debugging with compute-sanitizer

Each program here contains a classic CUDA bug next to its fix. By default the
fixed version runs and passes; `--buggy` runs the broken one, which may still
print the right answer, which is exactly why these bugs survive. The tool that
finds each one ships with the CUDA Toolkit:

| Example | Bug | Find it with |
| --- | --- | --- |
| [00_out_of_bounds](00_out_of_bounds/00_out_of_bounds.cu) | Off-by-one bounds check writes past the end of an allocation | `compute-sanitizer --tool memcheck ./00_out_of_bounds --buggy` |
| [01_shared_memory_race](01_shared_memory_race/01_shared_memory_race.cu) | Tree reduction in shared memory without `__syncthreads()` between levels | `compute-sanitizer --tool racecheck ./01_shared_memory_race --buggy` |
| [02_uninitialized_memory](02_uninitialized_memory/02_uninitialized_memory.cu) | Accumulating into `cudaMalloc` memory that was never cleared | `compute-sanitizer --tool initcheck ./02_uninitialized_memory --buggy` |
| [03_syncwarp_mask](03_syncwarp_mask/03_syncwarp_mask.cu) | `__syncwarp()` with a mask that leaves out lanes that call it | `compute-sanitizer --tool synccheck ./03_syncwarp_mask --buggy` |

The binaries are in `build/bin/debugging/`. The build adds `-lineinfo`, so the
reports point at source lines.

`make sanitize` (or `ctest -L 'memcheck|racecheck|initcheck|synccheck'`) runs
every example in the repo under memcheck and racecheck, and additionally runs
each buggy version here under its tool as a `.detects-bug` test, which passes
only if the sanitizer reports the error.
