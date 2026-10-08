/*
 * Pinned vs. Pageable Host Memory
 *
 * Intention:
 * Host memory from malloc/new is "pageable": the operating system may move
 * or swap its pages, so the GPU's copy engines cannot read it directly. A
 * cudaMemcpy from pageable memory is staged through a pinned (page-locked)
 * bounce buffer by the driver, which costs bandwidth. Memory allocated with
 * cudaMallocHost is pinned from the start, so the copy engine can DMA it
 * directly; it is also required for cudaMemcpyAsync to actually overlap with
 * other work (see 02_streams_overlap).
 *
 * High-Level Algorithm:
 * - Allocate the same amount of pageable and pinned host memory.
 * - Time host-to-device and device-to-host copies from each with CUDA events.
 * - Round-trip the data and check it is unchanged.
 *
 * Pinned memory is a limited resource (it cannot be paged out), so pin the
 * buffers you transfer often, not everything.
 */
#include <cstdlib>
#include <cstring>
#include <vector>

#include "lab.cuh"

float copy_ms(void *dst, const void *src, size_t bytes, cudaMemcpyKind kind) {
  return lab::time_ms([&] { CUDA_CHECK(cudaMemcpy(dst, src, bytes, kind)); }, 2, 10);
}

int main(int argc, char **argv) {
  lab::Args args(argc, argv);
  const size_t mb = static_cast<size_t>(args.get_int("mb", args.quick() ? 16 : 256));
  const size_t bytes = mb << 20;

  lab::print_device();
  printf("Copying %zu MB between host and device\n", mb);

  unsigned char *pageable = static_cast<unsigned char *>(std::malloc(bytes));
  unsigned char *pinned = nullptr;
  unsigned char *d_buf = nullptr;
  CUDA_CHECK(cudaMallocHost(&pinned, bytes));
  CUDA_CHECK(cudaMalloc(&d_buf, bytes));
  for (size_t i = 0; i < bytes; ++i) {
    pageable[i] = static_cast<unsigned char>(i * 7);
  }
  std::memcpy(pinned, pageable, bytes);

  lab::report("H2D pageable", copy_ms(d_buf, pageable, bytes, cudaMemcpyHostToDevice), 0, bytes);
  lab::report("H2D pinned", copy_ms(d_buf, pinned, bytes, cudaMemcpyHostToDevice), 0, bytes);
  lab::report("D2H pageable", copy_ms(pageable, d_buf, bytes, cudaMemcpyDeviceToHost), 0, bytes);
  lab::report("D2H pinned", copy_ms(pinned, d_buf, bytes, cudaMemcpyDeviceToHost), 0, bytes);
  printf("(%% of peak here is relative to GPU DRAM bandwidth; the real limit\n"
         " for these copies is the PCIe or NVLink connection.)\n");

  // Round trip: pageable -> device -> pinned must reproduce the data.
  for (size_t i = 0; i < bytes; ++i) {
    pageable[i] = static_cast<unsigned char>(i * 13 + 1);
  }
  CUDA_CHECK(cudaMemcpy(d_buf, pageable, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(pinned, d_buf, bytes, cudaMemcpyDeviceToHost));
  const bool pass = std::memcmp(pageable, pinned, bytes) == 0;
  printf("Check round trip               %s\n", pass ? "ok" : "FAILED");

  std::free(pageable);
  CUDA_CHECK(cudaFreeHost(pinned));
  CUDA_CHECK(cudaFree(d_buf));
  return lab::finish(pass);
}
