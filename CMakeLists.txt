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

# lab_example(<topic>/<...>/<name>.cu [LIBRARIES <targets>...])
#
# Builds one example into build/bin/<topic>/<name> and registers tests:
#   <topic>-<name>             label "run":       full-size run, must PASS
#   <topic>-<name>.memcheck    label "memcheck":  compute-sanitizer, --quick
#   <topic>-<name>.racecheck   label "racecheck": compute-sanitizer, --quick
function(lab_example source)
  cmake_parse_arguments(ARG "" "" "LIBRARIES" ${ARGN})

  get_filename_component(name "${source}" NAME_WE)
  string(REGEX MATCH "^[^/]+" topic "${source}")
  set(target "${topic}-${name}")

  add_executable(${target} "${source}")
  target_include_directories(${target} PRIVATE "${PROJECT_SOURCE_DIR}/common")
  # CMake links the CUDA runtime itself; only extra libraries are listed here.
  target_link_libraries(${target} PRIVATE ${ARG_LIBRARIES})
  # -lineinfo lets Nsight Compute map metrics back to source lines.
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  set_target_properties(${target} PROPERTIES
    OUTPUT_NAME "${name}"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin/${topic}")

  add_test(NAME ${target} COMMAND ${target})
  set_tests_properties(${target} PROPERTIES LABELS run)

  if(COMPUTE_SANITIZER)
    foreach(tool memcheck racecheck)
      add_test(NAME ${target}.${tool}
               COMMAND ${COMPUTE_SANITIZER} --tool ${tool} --error-exitcode 1
                       $<TARGET_FILE:${target}> --quick)
      set_tests_properties(${target}.${tool} PROPERTIES LABELS ${tool})
    endforeach()
  endif()
endfunction()

# basics
lab_example(basics/00_hello_warp_and_lane/00_hello_warp_and_lane.cu)
lab_example(basics/01_hello_thread_hierarchy/01_hello_thread_hierarchy.cu)
lab_example(basics/02_vector_add/02_vector_add.cu)
lab_example(basics/03_runtime_api_device_query/03_runtime_api_device_query.cu)

# memory
lab_example(memory/00_unified_memory_vector_add/00_unified_memory_vector_add.cu)

# warp
lab_example(warp/00_shuffle_intrinsics/00_shuffle_intrinsics.cu)

# reduction
lab_example(reduction/00_naive/00_reduction_naive.cu)
lab_example(reduction/01_shared/01_reduction_shared.cu)

# scan
lab_example(scan/00_kogge_stone/00_scan_kogge_stone.cu)
lab_example(scan/01_hillis_steele_double_buffer/01_scan_hillis_steele_double_buffer.cu)
lab_example(scan/02_brent_kung/02_scan_brent_kung.cu)
lab_example(scan/03_blelloch/03_scan_blelloch.cu)
lab_example(scan/04_blelloch_bank_conflict_free/04_scan_blelloch_bank_conflict_free.cu)
lab_example(scan/05_multiblock_cpu_fixup/05_scan_multiblock_cpu_fixup.cu)
lab_example(scan/06_multiblock_gpu_fixup/06_scan_multiblock_gpu_fixup.cu)

# matmul
lab_example(matmul/00_uncoalesced/00_matmul_uncoalesced.cu)
lab_example(matmul/01_coalesced/01_matmul_coalesced.cu)
lab_example(matmul/02_shared_memory/02_matmul_shared_memory.cu)
lab_example(matmul/03_register_tiling/03_matmul_register_tiling.cu)
lab_example(matmul/03_register_tiling/04_matmul_2d_register_tiling.cu)
lab_example(matmul/04_vectorized/04_matmul_vectorized.cu)
lab_example(matmul/05_tensor_cores/05_matmul_tensor_cores.cu)

# matrix_transpose
lab_example(matrix_transpose/00_transpose_bank_conflicts.cu)

# libraries
lab_example(libraries/00_cublas_gemm/00_cublas_gemm.cu LIBRARIES CUDA::cublas)

# sort
lab_example(sort/00_counting_sort/00_counting_sort.cu)
lab_example(sort/01_merge_sort/01_merge_sort.cu)
lab_example(sort/02_epsilon_sort_thrust/02_epsilon_sort_thrust.cu)
lab_example(sort/03_epsilon_sort_manual/03_epsilon_sort_manual.cu)

# sparse
lab_example(sparse/00_spmv_coo/00_spmv_coo.cu)
lab_example(sparse/01_spmv_csr/01_spmv_csr.cu)
lab_example(sparse/02_spmv_ell/02_spmv_ell.cu)
lab_example(sparse/03_spgemm_cusparse/03_spgemm_cusparse.cu LIBRARIES CUDA::cusparse)

# xor
lab_example(xor/00_set_symmetric_difference/00_xor_set_symmetric_difference.cu)

# optimization
lab_example(optimization/00_ant_colony_tsp/00_ant_colony_tsp.cu)
