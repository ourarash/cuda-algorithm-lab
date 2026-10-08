/*
 * CUDA Graphs: Removing Launch Overhead
 *
 * Intention:
 * Every kernel launch costs a few microseconds of CPU and driver work. For
 * big kernels that is noise, but workloads made of many small kernels (an
 * inference step, a solver iteration) can spend more time launching than
 * computing. A CUDA graph records a whole sequence of operations once and
 * replays it with a single launch, so the per-kernel overhead is paid once
 * at instantiation instead of on every iteration.
 *
 * High-Level Algorithm:
 * - One "iteration" is KERNELS small dependent kernels (each adds 1 to a
 *   vector of 4096 floats).
 * - Plain: launch the kernels one by one, ITERATIONS times.
 * - Graph: capture one iteration with cudaStreamBeginCapture /
 *   cudaStreamEndCapture (stream capture records instead of executing),
 *   instantiate it once, then cudaGraphLaunch it ITERATIONS times.
 * - Both must produce the same values (every element ends at
 *   KERNELS * ITERATIONS).
 */
#include <vector>

#include "lab.cuh"

constexpr int N = 4096;
constexpr int KERNELS = 20;

__global__ void add_one(float *x, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] += 1.0f;
}

int main(int argc, char **argv) {
  lab::Args args(argc, argv);
  const int iterations = static_cast<int>(args.get_int("iters", args.quick() ? 20 : 1000));

  lab::print_device();
  printf("%d iterations of %d tiny kernels (%d floats each)\n", iterations,
         KERNELS, N);

  float *d_x;
  CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));  // Capture needs a non-default stream

  auto one_iteration = [&] {
    for (int k = 0; k < KERNELS; ++k) {
      add_one<<<N / 256, 256, 0, stream>>>(d_x, N);
    }
  };

  // Record one iteration into a graph and instantiate it.
  cudaGraph_t graph;
  cudaGraphExec_t graph_exec;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  one_iteration();
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, 0));
  size_t num_nodes = 0;
  CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &num_nodes));
  printf("Captured graph with %zu kernel nodes\n", num_nodes);

  auto run_plain = [&] {
    for (int it = 0; it < iterations; ++it) one_iteration();
    CUDA_CHECK(cudaStreamSynchronize(stream));
  };
  auto run_graph = [&] {
    for (int it = 0; it < iterations; ++it) {
      CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
  };

  // Correctness: both versions add KERNELS * iterations to every element.
  const std::vector<float> expected(N, static_cast<float>(KERNELS * iterations));
  std::vector<float> got(N);
  bool pass = true;
  for (int version = 0; version < 2; ++version) {
    CUDA_CHECK(cudaMemset(d_x, 0, N * sizeof(float)));
    version == 0 ? run_plain() : run_graph();
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaMemcpy(got.data(), d_x, N * sizeof(float), cudaMemcpyDeviceToHost));
    pass = lab::check_equal(version == 0 ? "plain launches" : "graph launches", got,
                            expected) && pass;
  }

  const float plain_ms = lab::time_ms(run_plain, 1, 5);
  const float graph_ms = lab::time_ms(run_graph, 1, 5);
  lab::report("plain launches", plain_ms, 0, 0);
  lab::report("graph launches", graph_ms, 0, 0);
  printf("%-30s %9.2f us per kernel (plain), %.2f us (graph)\n", "Per kernel",
         1e3 * plain_ms / (iterations * KERNELS), 1e3 * graph_ms / (iterations * KERNELS));

  CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_x));
  return lab::finish(pass);
}
