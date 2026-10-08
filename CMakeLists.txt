cmake_minimum_required(VERSION 3.24)

# Build for the GPU in this machine by default. Override for other targets,
# for example: cmake -B build -DCMAKE_CUDA_ARCHITECTURES="80;90"
# (CUDA 13 supports compute capability 7.5 and newer.)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(cuda_algorithm_lab LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_program(COMPUTE_SANITIZER compute-sanitizer HINTS "${CUDAToolkit_BIN_DIR}")

enable_testing()

# Exit code an example returns when this GPU cannot run it (for example, a
# Hopper-only kernel on an Ampere GPU). ctest reports it as skipped.
set(LAB_SKIP_RETURN_CODE 77)

# lab_example(<topic>/<...>/<name>.cu [LIBRARIES <targets>...] [GENCODE <spec>]
#             [DETECT <sanitizer tool>])
#
# Builds one example into build/bin/<topic>/<name> and registers tests:
#   <topic>-<name>             label "run":       full-size run, must PASS
#   <topic>-<name>.memcheck    label "memcheck":  compute-sanitizer, --quick
#   <topic>-<name>.racecheck   label "racecheck": compute-sanitizer, --quick
# GENCODE replaces CMAKE_CUDA_ARCHITECTURES for examples that need an
# architecture-specific target such as "arch=compute_90a,code=sm_90a".
# DETECT (debugging/ examples) adds <topic>-<name>.detects-bug, which runs the
# example's deliberately buggy version (--buggy) under that compute-sanitizer
# tool and passes only if the tool reports the bug.
function(lab_example source)
  cmake_parse_arguments(ARG "" "GENCODE;DETECT" "LIBRARIES" ${ARGN})

  get_filename_component(name "${source}" NAME_WE)
  string(REGEX MATCH "^[^/]+" topic "${source}")
  set(target "${topic}-${name}")

  add_executable(${target} "${source}")
  target_include_directories(${target} PRIVATE "${PROJECT_SOURCE_DIR}/common")
  # CMake links the CUDA runtime itself; only extra libraries are listed here.
  target_link_libraries(${target} PRIVATE ${ARG_LIBRARIES})
  # -lineinfo lets Nsight Compute map metrics back to source lines.
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  if(ARG_GENCODE)
    set_target_properties(${target} PROPERTIES CUDA_ARCHITECTURES OFF)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-gencode=${ARG_GENCODE}>)
  endif()
  set_target_properties(${target} PROPERTIES
    OUTPUT_NAME "${name}"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin/${topic}")

  add_test(NAME ${target} COMMAND ${target})
  set_tests_properties(${target} PROPERTIES
    LABELS run SKIP_RETURN_CODE ${LAB_SKIP_RETURN_CODE})

  if(COMPUTE_SANITIZER)
    foreach(tool memcheck racecheck)
      add_test(NAME ${target}.${tool}
               COMMAND ${COMPUTE_SANITIZER} --tool ${tool} --error-exitcode 1
                       $<TARGET_FILE:${target}> --quick)
      set_tests_properties(${target}.${tool} PROPERTIES
        LABELS ${tool} SKIP_RETURN_CODE ${LAB_SKIP_RETURN_CODE})
    endforeach()
    if(ARG_DETECT)
      add_test(NAME ${target}.detects-bug
               COMMAND ${COMPUTE_SANITIZER} --tool ${ARG_DETECT}
                       $<TARGET_FILE:${target}> --buggy)
      set_tests_properties(${target}.detects-bug PROPERTIES
        LABELS ${ARG_DETECT}
        PASS_REGULAR_EXPRESSION "ERROR SUMMARY: [1-9];RACECHECK SUMMARY: [1-9]")
    endif()
  endif()
endfunction()

# basics
lab_example(basics/00_hello_warp_and_lane/00_hello_warp_and_lane.cu)
lab_example(basics/01_hello_thread_hierarchy/01_hello_thread_hierarchy.cu)
lab_example(basics/02_vector_add/02_vector_add.cu)
lab_example(basics/03_runtime_api_device_query/03_runtime_api_device_query.cu)
lab_example(basics/04_occupancy_launch_bounds/04_occupancy_launch_bounds.cu)
lab_example(basics/05_warp_divergence/05_warp_divergence.cu)
lab_example(basics/06_cooperative_groups/06_cooperative_groups.cu)

# memory
lab_example(memory/00_unified_memory_vector_add/00_unified_memory_vector_add.cu)
lab_example(memory/01_pinned_vs_pageable/01_pinned_vs_pageable.cu)
lab_example(memory/02_streams_overlap/02_streams_overlap.cu)
lab_example(memory/03_cuda_graphs/03_cuda_graphs.cu)
lab_example(memory/04_unified_memory_prefetch/04_unified_memory_prefetch.cu)

# warp
lab_example(warp/00_shuffle_intrinsics/00_shuffle_intrinsics.cu)

# reduction
lab_example(reduction/00_naive/00_reduction_naive.cu)
lab_example(reduction/01_interleaved_divergent/01_reduction_interleaved_divergent.cu)
lab_example(reduction/02_interleaved_bank_conflicts/02_reduction_interleaved_bank_conflicts.cu)
lab_example(reduction/03_sequential_addressing/03_reduction_sequential_addressing.cu)
lab_example(reduction/04_first_add_during_load/04_reduction_first_add_during_load.cu)
lab_example(reduction/05_unroll_last_warp/05_reduction_unroll_last_warp.cu)
lab_example(reduction/06_complete_unroll/06_reduction_complete_unroll.cu)
lab_example(reduction/07_grid_stride/07_reduction_grid_stride.cu)
lab_example(reduction/08_warp_shuffle/08_reduction_warp_shuffle.cu)
lab_example(reduction/09_cooperative_groups/09_reduction_cooperative_groups.cu)
lab_example(reduction/10_single_pass/10_reduction_single_pass.cu)
lab_example(reduction/11_vectorized/11_reduction_vectorized.cu)
lab_example(reduction/12_cub/12_reduction_cub.cu)

# scan
lab_example(scan/00_kogge_stone/00_scan_kogge_stone.cu)
lab_example(scan/01_hillis_steele_double_buffer/01_scan_hillis_steele_double_buffer.cu)
lab_example(scan/02_brent_kung/02_scan_brent_kung.cu)
lab_example(scan/03_blelloch/03_scan_blelloch.cu)
lab_example(scan/04_blelloch_bank_conflict_free/04_scan_blelloch_bank_conflict_free.cu)
lab_example(scan/05_multiblock_cpu_fixup/05_scan_multiblock_cpu_fixup.cu)
lab_example(scan/06_multiblock_gpu_fixup/06_scan_multiblock_gpu_fixup.cu)
lab_example(scan/07_warp_shuffle_reduce_then_scan/07_scan_warp_shuffle.cu)
lab_example(scan/08_decoupled_lookback/08_scan_decoupled_lookback.cu)
lab_example(scan/09_cub/09_scan_cub.cu)

# histogram
lab_example(histogram/00_global_atomics/00_histogram_global_atomics.cu)
lab_example(histogram/01_shared_privatization/01_histogram_shared_privatization.cu)
lab_example(histogram/02_aggregation/02_histogram_aggregation.cu)
lab_example(histogram/03_cub/03_histogram_cub.cu)

# compaction
lab_example(compaction/00_scan_based/00_compaction_scan_based.cu)
lab_example(compaction/01_warp_aggregated_atomics/01_compaction_warp_aggregated.cu)
lab_example(compaction/02_cub/02_compaction_cub.cu)

# matmul (every stage prints cuBLAS as its baseline)
lab_example(matmul/00_uncoalesced/00_matmul_uncoalesced.cu LIBRARIES CUDA::cublas)
lab_example(matmul/01_coalesced/01_matmul_coalesced.cu LIBRARIES CUDA::cublas)
lab_example(matmul/02_shared_memory/02_matmul_shared_memory.cu LIBRARIES CUDA::cublas)
lab_example(matmul/03_1d_register_tiling/03_matmul_1d_register_tiling.cu LIBRARIES CUDA::cublas)
lab_example(matmul/04_2d_register_tiling/04_matmul_2d_register_tiling.cu LIBRARIES CUDA::cublas)
lab_example(matmul/05_vectorized/05_matmul_vectorized.cu LIBRARIES CUDA::cublas)
lab_example(matmul/06_warptiling/06_matmul_warptiling.cu LIBRARIES CUDA::cublas)
lab_example(matmul/07_double_buffering/07_matmul_double_buffering.cu LIBRARIES CUDA::cublas)
lab_example(matmul/08_tensor_cores_wmma/08_matmul_wmma.cu LIBRARIES CUDA::cublas)
lab_example(matmul/09_wmma_shared_memory/09_matmul_wmma_shared_memory.cu LIBRARIES CUDA::cublas)
lab_example(matmul/10_mma_sync/10_matmul_mma_sync.cu LIBRARIES CUDA::cublas)
# Hopper wgmma/TMA code must be compiled for the architecture-specific
# sm_90a target, which CUDA 12.0 introduced.
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
  lab_example(matmul/11_hopper_wgmma_tma/11_matmul_hopper_wgmma.cu
              GENCODE arch=compute_90a,code=sm_90a
              LIBRARIES CUDA::cublas CUDA::cuda_driver)
endif()

# matrix_transpose
lab_example(matrix_transpose/00_copy/00_transpose_copy.cu)
lab_example(matrix_transpose/01_naive/01_transpose_naive.cu)
lab_example(matrix_transpose/02_shared_memory/02_transpose_shared_memory.cu)
lab_example(matrix_transpose/03_padded/03_transpose_padded.cu)
lab_example(matrix_transpose/04_swizzled/04_transpose_swizzled.cu)

# sort
lab_example(sort/00_counting_sort/00_counting_sort.cu)
lab_example(sort/01_merge_sort/01_merge_sort.cu)
lab_example(sort/02_bitonic_sort/02_bitonic_sort.cu)
lab_example(sort/03_radix_sort/03_radix_sort.cu)
lab_example(sort/04_cub_radix_sort/04_cub_radix_sort.cu)

# convolution and stencil
lab_example(convolution/00_naive/00_convolution_naive.cu)
lab_example(convolution/01_constant_memory/01_convolution_constant_memory.cu)
lab_example(convolution/02_shared_memory_tiles/02_convolution_shared_tiles.cu)
lab_example(stencil/00_naive/00_stencil_naive.cu)
lab_example(stencil/01_shared_memory_z_streaming/01_stencil_z_streaming.cu)

# ml (the PyTorch extension in ml/pytorch_extension builds with PyTorch)
lab_example(ml/00_softmax_naive/00_softmax_naive.cu)
lab_example(ml/01_softmax_online/01_softmax_online.cu)
lab_example(ml/02_layernorm/02_layernorm.cu)
lab_example(ml/03_rmsnorm/03_rmsnorm.cu)
lab_example(ml/04_attention_naive/04_attention_naive.cu)
lab_example(ml/05_flash_attention/05_flash_attention.cu)

# debugging (each has a deliberate bug behind --buggy; see debugging/README.md)
lab_example(debugging/00_out_of_bounds/00_out_of_bounds.cu DETECT memcheck)
lab_example(debugging/01_shared_memory_race/01_shared_memory_race.cu DETECT racecheck)
lab_example(debugging/02_uninitialized_memory/02_uninitialized_memory.cu DETECT initcheck)
lab_example(debugging/03_syncwarp_mask/03_syncwarp_mask.cu DETECT synccheck)

# sparse
lab_example(sparse/00_spmv_coo/00_spmv_coo.cu)
lab_example(sparse/01_spmv_csr/01_spmv_csr.cu)
lab_example(sparse/02_spmv_ell/02_spmv_ell.cu)
lab_example(sparse/03_spgemm_cusparse/03_spgemm_cusparse.cu LIBRARIES CUDA::cusparse)
lab_example(sparse/04_spmv_csr_scalar_power_law/04_spmv_csr_scalar_power_law.cu)
lab_example(sparse/05_spmv_csr_vector/05_spmv_csr_vector.cu)
lab_example(sparse/06_spmv_hybrid/06_spmv_hybrid.cu)
lab_example(sparse/07_spmv_cusparse/07_spmv_cusparse.cu LIBRARIES CUDA::cusparse)

# libraries
lab_example(libraries/00_cublas_gemm/00_cublas_gemm.cu LIBRARIES CUDA::cublas)
lab_example(libraries/01_thrust_set_symmetric_difference/01_thrust_set_symmetric_difference.cu)

# applications
lab_example(applications/00_ant_colony_tsp/00_ant_colony_tsp.cu)
lab_example(applications/01_epsilon_sort_thrust/01_epsilon_sort_thrust.cu)
lab_example(applications/02_epsilon_sort_manual/02_epsilon_sort_manual.cu)
