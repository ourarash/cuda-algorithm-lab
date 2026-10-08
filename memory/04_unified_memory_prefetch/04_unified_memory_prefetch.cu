/*
 * Unified Memory: Page Faults vs. Prefetching
 *
 * Intention:
 * Managed memory (cudaMallocManaged) migrates pages between CPU and GPU on
 * demand. On GPUs that support it (concurrentManagedAccess, Linux), the first
 * GPU access to a page that lives on the CPU raises a page fault, the page is
 * migrated, and the kernel continues. That is convenient but slow: faults are
 * handled in small batches while the kernel stalls. cudaMemPrefetchAsync
 * migrates the data in bulk before the kernel needs it.
 *
 * High-Level Algorithm (each case starts with the data freshly written by
 * the CPU, so it lives in host memory):
 * 1. on-demand: launch the kernel and let it fault the pages in;
 * 2. prefetch:  cudaMemPrefetchAsync to the GPU, then launch.
 * The kernel time in case 1 includes the migration; in case 2 the prefetch is
 * timed separately, and the kernel itself runs at full speed.
 *
 * API note: CUDA 13 changed cudaMemPrefetchAsync to take a cudaMemLocation
 * (device or host, plus a flags argument) instead of a device number; the
 * code below handles both versions.
 */
#include <vector>

#include "lab.cuh"

__global__ void scale_add(float *x, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = 2.0f * x[i] + 1.0f;
}

void prefetch_to_device(const float *p, size_t bytes, int device) {
#if CUDART_VERSION >= 13000
  cudaMemLocation location = {};
  location.type = cudaMemLocationTypeDevice;
  location.id = device;
  CUDA_CHECK(cudaMemPrefetchAsync(p, bytes, location, 0, 0));
#else
  CUDA_CHECK(cudaMemPrefetchAsync(p, bytes, device, 0));
#endif
}

float elapsed_ms(cudaEvent_t a, cudaEvent_t b) {
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  return ms;
}

int main(int argc, char **argv) {
  lab::Args args(argc, argv);
  const int n = static_cast<int>(args.get_int("n", args.quick() ? (1 << 16) : (1 << 26)));
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);

  lab::print_device();
  int device = 0;
  int concurrent = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaDeviceGetAttribute(&concurrent, cudaDevAttrConcurrentManagedAccess, device));
  printf("%d managed floats (%zu MB); GPU page faulting: %s\n", n, bytes >> 20,
         concurrent ? "yes" : "no");
  if (!concurrent) {
    return lab::skip("on-demand migration and prefetching need concurrentManagedAccess "
                     "(not available on Windows or this GPU)");
  }

  float *x;
  CUDA_CHECK(cudaMallocManaged(&x, bytes));
  cudaEvent_t e0, e1, e2;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  CUDA_CHECK(cudaEventCreate(&e2));
  const int blocks = lab::ceil_div(n, 256);
  bool pass = true;
  std::vector<float> got(n), expected(n);

  for (int use_prefetch = 0; use_prefetch < 2; ++use_prefetch) {
    for (int i = 0; i < n; ++i) x[i] = static_cast<float>(i % 100);  // Pages now on the CPU
    for (int i = 0; i < n; ++i) expected[i] = 2.0f * (i % 100) + 1.0f;

    CUDA_CHECK(cudaEventRecord(e0));
    if (use_prefetch) prefetch_to_device(x, bytes, device);
    CUDA_CHECK(cudaEventRecord(e1));
    scale_add<<<blocks, 256>>>(x, n);
    CUDA_CHECK(cudaEventRecord(e2));
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaEventSynchronize(e2));

    if (use_prefetch) {
      lab::report("prefetch (bulk migration)", elapsed_ms(e0, e1), 0, bytes);
      lab::report("kernel after prefetch", elapsed_ms(e1, e2), 0, 2.0 * bytes);
    } else {
      lab::report("kernel with page faults", elapsed_ms(e1, e2), 0, 2.0 * bytes);
    }
    for (int i = 0; i < n; ++i) got[i] = x[i];  // Reading on the CPU migrates back
    pass = lab::check_equal(use_prefetch ? "prefetch result" : "on-demand result",
                            got, expected) && pass;
  }

  CUDA_CHECK(cudaEventDestroy(e0));
  CUDA_CHECK(cudaEventDestroy(e1));
  CUDA_CHECK(cudaEventDestroy(e2));
  CUDA_CHECK(cudaFree(x));
  return lab::finish(pass);
}
