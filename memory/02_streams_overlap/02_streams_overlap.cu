/*
 * Overlapping Transfers and Compute with Streams
 *
 * Intention:
 * A GPU can copy to the device, compute, and copy back at the same time: the
 * copy engines are independent of the SMs. Work in the same CUDA stream runs
 * in order, but work in different streams may overlap. Splitting a job into
 * chunks and giving each chunk its own copy-compute-copy sequence in one of
 * several streams turns
 *     [  H2D all  ][ kernel all ][  D2H all  ]
 * into a pipeline where chunk i's kernel runs while chunk i+1 is copied in
 * and chunk i-1 is copied out.
 *
 * High-Level Algorithm:
 * - Sequential: copy the whole input in, run the kernel, copy the result out.
 * - Overlapped: CHUNKS pieces, round-robin over STREAMS streams, each with
 *   cudaMemcpyAsync H2D, kernel, cudaMemcpyAsync D2H in its stream.
 * - Both must produce the same output; the timings show the gain.
 *
 * Requirements for overlap: pinned host memory (cudaMallocHost), async
 * copies, non-default streams, and a GPU with at least one copy engine
 * (asyncEngineCount, printed below). The kernel does enough math per element
 * that compute and transfer times are comparable, which is when overlap pays.
 */
#include <cmath>
#include <vector>

#include "lab.cuh"

constexpr int STREAMS = 4;
constexpr int CHUNKS = 16;

__global__ void work(const float *in, float *out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = in[i];
    for (int k = 0; k < 64; ++k) {
      x = sinf(x) * 0.5f + cosf(x) * 0.5f;
    }
    out[i] = x;
  }
}

int main(int argc, char **argv) {
  lab::Args args(argc, argv);
  const int n = static_cast<int>(args.get_int("n", args.quick() ? (1 << 16) : (1 << 25)));
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);

  lab::print_device();
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("%d floats, %d chunks over %d streams; copy engines: %d\n", n, CHUNKS,
         STREAMS, prop.asyncEngineCount);

  float *h_in, *h_out_seq, *h_out_ovl;
  CUDA_CHECK(cudaMallocHost(&h_in, bytes));
  CUDA_CHECK(cudaMallocHost(&h_out_seq, bytes));
  CUDA_CHECK(cudaMallocHost(&h_out_ovl, bytes));
  for (int i = 0; i < n; ++i) {
    h_in[i] = static_cast<float>(i % 1000) * 0.001f;
  }
  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  cudaStream_t streams[STREAMS];
  for (auto &s : streams) {
    CUDA_CHECK(cudaStreamCreate(&s));
  }

  auto sequential = [&] {
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    work<<<lab::ceil_div(n, 256), 256>>>(d_in, d_out, n);
    CUDA_CHECK(cudaMemcpy(h_out_seq, d_out, bytes, cudaMemcpyDeviceToHost));
  };
  auto overlapped = [&] {
    const int chunk = lab::ceil_div(n, CHUNKS);
    for (int c = 0; c < CHUNKS; ++c) {
      const int offset = c * chunk;
      const int count = std::min(chunk, n - offset);
      if (count <= 0) break;
      cudaStream_t s = streams[c % STREAMS];
      CUDA_CHECK(cudaMemcpyAsync(d_in + offset, h_in + offset,
                                 count * sizeof(float), cudaMemcpyHostToDevice, s));
      work<<<lab::ceil_div(count, 256), 256, 0, s>>>(d_in + offset, d_out + offset, count);
      CUDA_CHECK(cudaMemcpyAsync(h_out_ovl + offset, d_out + offset,
                                 count * sizeof(float), cudaMemcpyDeviceToHost, s));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
  };

  sequential();
  overlapped();
  CUDA_CHECK_LAUNCH();
  std::vector<float> a(h_out_seq, h_out_seq + n), b(h_out_ovl, h_out_ovl + n);
  const bool pass = lab::check_equal("overlapped == sequential", b, a);

  const float seq_ms = lab::time_ms(sequential, 1, 5);
  const float ovl_ms = lab::time_ms(overlapped, 1, 5);
  lab::report("sequential", seq_ms, 0, 0);
  lab::report("overlapped (streams)", ovl_ms, 0, 0);
  printf("%-30s %9.2f x\n", "Speedup", seq_ms / ovl_ms);

  for (auto &s : streams) {
    CUDA_CHECK(cudaStreamDestroy(s));
  }
  CUDA_CHECK(cudaFreeHost(h_in));
  CUDA_CHECK(cudaFreeHost(h_out_seq));
  CUDA_CHECK(cudaFreeHost(h_out_ovl));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return lab::finish(pass);
}
