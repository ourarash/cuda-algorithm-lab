/*
 * Ant Colony Optimization For TSP
 *
 * Intention:
 * This file is a toy CUDA implementation of ant colony optimization for the
 * travelling salesman problem. It is not meant to be a production solver; it
 * is meant to show how many candidate tours can be explored in parallel.
 *
 * High-Level Algorithm:
 * - Keep the distance matrix and pheromone matrix on the device.
 * - Launch one thread per ant so each thread builds one full tour.
 * - Measure each tour length and track the best one found so far.
 * - Evaporate pheromones globally, then reinforce edges from the best tour.
 * - Repeat for many iterations.
 */
#include <cuda.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 100     // number of cities
#define THREADS N // one thread per ant
#define BLOCKS 1
#define MAX_ITERS 1000 // number of ACO iterations
#define Q 100.0f       // pheromone deposit constant
#define RHO 0.1f       // evaporation rate

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
  if ((call) != cudaSuccess) {                                                 \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,           \
            cudaGetErrorString(cudaGetLastError()));                           \
    exit(EXIT_FAILURE);                                                        \
  }

// Global device memory
__device__ float d_dist[N][N];          // Distance matrix
__device__ float d_pheromone[N][N];     // Pheromone matrix
__device__ int d_best_path[N];          // Best tour path found so far
__device__ float d_best_length = 1e30f; // Best tour length (initialized high)

// Initialize random number generator (one per thread)
__global__ void init_curand(curandState *states, unsigned long seed) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(seed, id, 0, &states[id]);
}

// Compute total length of a given tour
__device__ float compute_length(int *tour) {
  float sum = 0;
  for (int i = 0; i < N; ++i)
    sum += d_dist[tour[i]][tour[(i + 1) % N]];
  return sum;
}

// Kernel: each thread builds a full tour (one ant per thread)
__global__ void ant_colony_kernel(curandState *states) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= N)
    return;

  int tour[N];           // tour built by this thread
  bool visited[N] = {0}; // track visited cities
  curandState local_state = states[tid];

  // Start tour from the thread's assigned city
  int city = tid;
  tour[0] = city;
  visited[city] = true;

  // Build the tour step by step
  for (int step = 1; step < N; ++step) {
    float prob[N] = {0}; // edge selection probabilities
    float sum = 0.0f;

    // Compute unnormalized probabilities
    for (int j = 0; j < N; ++j) {
      if (!visited[j]) {
        float tau = d_pheromone[city][j];             // pheromone
        float eta = 1.0f / (d_dist[city][j] + 1e-6f); // inverse distance
        prob[j] = tau * eta;
        sum += prob[j];
      }
    }

    // Sample next city using roulette wheel selection
    float r = curand_uniform(&local_state) * sum;
    float acc = 0.0f;
    int next_city = -1;

    for (int j = 0; j < N; ++j) {
      if (!visited[j]) {
        acc += prob[j];
        if (acc >= r) {
          next_city = j;
          break;
        }
      }
    }

    city = next_city;
    tour[step] = city;
    visited[city] = true;
  }

  // Compute tour length
  float L = compute_length(tour);

  // Atomically update global best if this tour is better
  if (atomicMin(&d_best_length, L) > L) {
    for (int i = 0; i < N; ++i)
      d_best_path[i] = tour[i];
  }

  // Save RNG state for next iteration
  states[tid] = local_state;
}

// Kernel: evaporate and reinforce pheromones based on best tour
__global__ void update_pheromones_kernel(float Q, float rho, int *best_path,
                                         float best_length) {
  int i = threadIdx.x;
  int j = threadIdx.y;
  if (i >= N || j >= N)
    return;

  // Evaporate pheromone on all edges
  d_pheromone[i][j] *= (1.0f - rho);

  // Reinforce edges in best path only
  for (int k = 0; k < N; ++k) {
    int from = best_path[k];
    int to = best_path[(k + 1) % N]; // wrap around to starting city
    if ((i == from && j == to) || (i == to && j == from)) {
      d_pheromone[i][j] += Q / best_length;
    }
  }
}

// Host-side driver for the full ant-colony optimization loop.
int main() {
  float h_dist[N][N], h_pheromone[N][N];
  int h_best_path[N];
  float h_best_length;

  // Initialize host matrices
  srand(time(NULL));
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      h_dist[i][j] =
          (i == j) ? 1e6 : (float)(rand() % 100 + 1); // avoid 0 distance
      h_pheromone[i][j] = 1.0f; // uniform initial pheromone
    }

  // Copy data to device memory
  CUDA_CHECK(cudaMemcpyToSymbol(d_dist, h_dist, sizeof(float) * N * N));
  CUDA_CHECK(
      cudaMemcpyToSymbol(d_pheromone, h_pheromone, sizeof(float) * N * N));

  // Allocate and initialize curand RNG states
  curandState *d_states;
  CUDA_CHECK(cudaMalloc(&d_states, N * sizeof(curandState)));
  init_curand<<<1, THREADS>>>(d_states, time(NULL));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Get pointer to d_best_path
  int *d_best_path_ptr;
  CUDA_CHECK(cudaGetSymbolAddress((void **)&d_best_path_ptr, d_best_path));

  // Main optimization loop
  for (int iter = 0; iter < MAX_ITERS; ++iter) {
    // Construct tours
    ant_colony_kernel<<<BLOCKS, THREADS>>>(d_states);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Get best length so far
    CUDA_CHECK(
        cudaMemcpyFromSymbol(&h_best_length, d_best_length, sizeof(float)));

    // Update pheromones based on best tour
    update_pheromones_kernel<<<1, dim3(N, N)>>>(Q, RHO, d_best_path_ptr,
                                                h_best_length);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Copy result back
  CUDA_CHECK(
      cudaMemcpyFromSymbol(&h_best_length, d_best_length, sizeof(float)));
  CUDA_CHECK(cudaMemcpyFromSymbol(h_best_path, d_best_path, sizeof(int) * N));

  // Print best tour and length
  printf("Best tour length: %.2f\nPath:\n", h_best_length);
  for (int i = 0; i < N; ++i)
    printf("%d ", h_best_path[i]);
  printf("\n");

  cudaFree(d_states);
  return 0;
}
