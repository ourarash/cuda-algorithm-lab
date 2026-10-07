/*
 * Ant Colony Optimization For TSP
 *
 * Intention:
 * This file is a toy CUDA implementation of ant colony optimization for the
 * travelling salesman problem. It is not meant to be a production solver; it
 * is meant to show how many candidate tours can be explored in parallel, and
 * how to combine per-thread results without races.
 *
 * High-Level Algorithm (one iteration):
 * 1. Construct: one thread per ant builds a full tour. At each step it picks
 *    the next unvisited city with probability proportional to
 *    pheromone(edge) / distance(edge). Each ant writes its tour and length to
 *    its own slot in global memory, so no two threads write the same place.
 * 2. Select: a separate kernel finds the shortest tour of this iteration and,
 *    if it beats the best so far, copies it into the global best.
 * 3. Evaporate: one thread per matrix entry scales every pheromone by
 *    (1 - rho), using a 2D grid of 16 x 16 blocks.
 * 4. Deposit: one thread per edge of the best tour adds Q / best_length to
 *    that edge in both directions.
 *
 * Why the select step is separate:
 * A tempting shortcut is to let every ant do "if my tour is better than the
 * best, copy it into best_path" with an atomic min on the length. But the
 * atomic only protects the length. Two ants that both improve on the old best
 * can interleave their copies, leaving best_path as a mix of two tours that
 * is not even a valid permutation. Writing results to separate slots and
 * choosing afterwards avoids that race entirely.
 *
 * The cities are random points in a square, so distances are symmetric and
 * Euclidean. The program checks that the best tour visits every city exactly
 * once and that its reported length is right, and prints a greedy
 * nearest-neighbor tour for comparison.
 */
#include <curand_kernel.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "lab.cuh"

constexpr int kNumCities = 100;
constexpr int kNumAnts = kNumCities;  // One ant per starting city
constexpr float kDeposit = 100.0f;    // Q: pheromone deposited per tour
constexpr float kEvaporation = 0.1f;  // rho: fraction evaporated per iteration

// Distance and pheromone matrices live in global memory as __device__
// variables, filled from the host with cudaMemcpyToSymbol.
__device__ float d_dist[kNumCities][kNumCities];
__device__ float d_pheromone[kNumCities][kNumCities];

// One random-number generator state per ant.
__global__ void init_curand(curandState *states, unsigned long long seed) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < kNumAnts) {
    curand_init(seed, id, 0, &states[id]);
  }
}

// 1. Construct: each thread builds one tour.
__global__ void construct_tours_kernel(curandState *states, int *tours,
                                       float *lengths) {
  int ant = threadIdx.x + blockIdx.x * blockDim.x;
  if (ant >= kNumAnts) {
    return;
  }

  // Per-thread arrays this large do not fit in registers, so the compiler
  // places them in (cached) local memory. That is acceptable for a toy.
  int tour[kNumCities];
  bool visited[kNumCities] = {};
  float prob[kNumCities];
  curandState local_state = states[ant];

  int city = ant % kNumCities;
  tour[0] = city;
  visited[city] = true;

  for (int step = 1; step < kNumCities; ++step) {
    // Unnormalized probability of moving to each unvisited city.
    float sum = 0.0f;
    for (int j = 0; j < kNumCities; ++j) {
      prob[j] = 0.0f;
      if (!visited[j]) {
        float tau = d_pheromone[city][j];
        float eta = 1.0f / (d_dist[city][j] + 1e-6f);
        prob[j] = tau * eta;
        sum += prob[j];
      }
    }

    // Roulette-wheel selection. Start from the last unvisited city so a
    // rounding shortfall in the running sum still yields a valid choice.
    float r = curand_uniform(&local_state) * sum;
    float acc = 0.0f;
    int next_city = -1;
    for (int j = 0; j < kNumCities; ++j) {
      if (!visited[j]) {
        next_city = j;
        acc += prob[j];
        if (acc >= r) {
          break;
        }
      }
    }

    city = next_city;
    tour[step] = city;
    visited[city] = true;
  }

  float length = 0.0f;
  for (int i = 0; i < kNumCities; ++i) {
    length += d_dist[tour[i]][tour[(i + 1) % kNumCities]];
  }

  for (int i = 0; i < kNumCities; ++i) {
    tours[ant * kNumCities + i] = tour[i];
  }
  lengths[ant] = length;
  states[ant] = local_state;
}

// 2. Select: keep the best tour seen so far. With only kNumAnts candidates a
// single thread is plenty; a large colony would use a parallel argmin
// reduction (see reduction/).
__global__ void select_best_kernel(const int *tours, const float *lengths,
                                   int *best_tour, float *best_length) {
  int best_ant = 0;
  for (int ant = 1; ant < kNumAnts; ++ant) {
    if (lengths[ant] < lengths[best_ant]) {
      best_ant = ant;
    }
  }
  if (lengths[best_ant] < *best_length) {
    *best_length = lengths[best_ant];
    for (int i = 0; i < kNumCities; ++i) {
      best_tour[i] = tours[best_ant * kNumCities + i];
    }
  }
}

// 3. Evaporate: one thread per pheromone entry.
__global__ void evaporate_kernel(float rho) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < kNumCities && j < kNumCities) {
    d_pheromone[i][j] *= (1.0f - rho);
  }
}

// 4. Deposit: one thread per edge of the best tour. A tour visits each city
// once, so every undirected edge appears at most once and no two threads
// update the same entry.
__global__ void deposit_kernel(float deposit, const int *best_tour,
                               const float *best_length) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < kNumCities) {
    int from = best_tour[k];
    int to = best_tour[(k + 1) % kNumCities];
    float amount = deposit / *best_length;
    d_pheromone[from][to] += amount;
    d_pheromone[to][from] += amount;
  }
}

// Greedy nearest-neighbor tour from city 0, as a CPU baseline.
double nearest_neighbor_length(const std::vector<float> &dist) {
  std::vector<bool> visited(kNumCities, false);
  int city = 0;
  visited[0] = true;
  double length = 0.0;
  for (int step = 1; step < kNumCities; ++step) {
    int next = -1;
    for (int j = 0; j < kNumCities; ++j) {
      if (!visited[j] &&
          (next < 0 || dist[city * kNumCities + j] < dist[city * kNumCities + next])) {
        next = j;
      }
    }
    length += dist[city * kNumCities + next];
    visited[next] = true;
    city = next;
  }
  return length + dist[city * kNumCities + 0];
}

int main(int argc, char **argv) {
  lab::Args args(argc, argv);
  const int iterations = static_cast<int>(args.get_int("iters", args.quick() ? 20 : 1000));

  lab::print_device();
  printf("Ant colony optimization: %d cities, %d ants, %d iterations\n",
         kNumCities, kNumAnts, iterations);

  // Random cities in a 100 x 100 square; symmetric Euclidean distances.
  std::mt19937 gen(2024);
  std::uniform_real_distribution<float> coord(0.0f, 100.0f);
  std::vector<float> x(kNumCities), y(kNumCities);
  for (int i = 0; i < kNumCities; ++i) {
    x[i] = coord(gen);
    y[i] = coord(gen);
  }
  std::vector<float> h_dist(kNumCities * kNumCities);
  for (int i = 0; i < kNumCities; ++i) {
    for (int j = 0; j < kNumCities; ++j) {
      h_dist[i * kNumCities + j] = std::hypot(x[i] - x[j], y[i] - y[j]);
    }
  }
  const std::vector<float> h_pheromone(kNumCities * kNumCities, 1.0f);
  CUDA_CHECK(cudaMemcpyToSymbol(d_dist, h_dist.data(),
                                h_dist.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_pheromone, h_pheromone.data(),
                                h_pheromone.size() * sizeof(float)));

  curandState *d_states;
  int *d_tours, *d_best_tour;
  float *d_lengths, *d_best_length;
  CUDA_CHECK(cudaMalloc(&d_states, kNumAnts * sizeof(curandState)));
  CUDA_CHECK(cudaMalloc(&d_tours, kNumAnts * kNumCities * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_lengths, kNumAnts * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_best_tour, kNumCities * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_best_length, sizeof(float)));
  const float initial_best = FLT_MAX;
  CUDA_CHECK(cudaMemcpy(d_best_length, &initial_best, sizeof(float),
                        cudaMemcpyHostToDevice));

  const int ant_threads = 128;
  const int ant_blocks = lab::ceil_div(kNumAnts, ant_threads);
  init_curand<<<ant_blocks, ant_threads>>>(d_states, /*seed=*/1234);
  CUDA_CHECK_LAUNCH();

  // A 2D launch for the N x N pheromone matrix. A single block of N x N
  // threads would need 10,000 threads, far above the 1,024-per-block limit.
  const dim3 evap_block(16, 16);
  const dim3 evap_grid(lab::ceil_div(kNumCities, 16), lab::ceil_div(kNumCities, 16));

  for (int iter = 0; iter < iterations; ++iter) {
    construct_tours_kernel<<<ant_blocks, ant_threads>>>(d_states, d_tours,
                                                        d_lengths);
    select_best_kernel<<<1, 1>>>(d_tours, d_lengths, d_best_tour, d_best_length);
    evaporate_kernel<<<evap_grid, evap_block>>>(kEvaporation);
    deposit_kernel<<<lab::ceil_div(kNumCities, 128), 128>>>(kDeposit, d_best_tour,
                                                           d_best_length);
    CUDA_CHECK_LAUNCH();
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  float best_length = 0.0f;
  std::vector<int> best_tour(kNumCities);
  CUDA_CHECK(cudaMemcpy(&best_length, d_best_length, sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(best_tour.data(), d_best_tour, kNumCities * sizeof(int),
                        cudaMemcpyDeviceToHost));

  printf("Best tour length (ACO):          %.2f\n", best_length);
  printf("Greedy nearest-neighbor length:  %.2f\n", nearest_neighbor_length(h_dist));
  printf("Best tour:");
  for (int city : best_tour) {
    printf(" %d", city);
  }
  printf("\n");

  // The tour must visit every city exactly once ...
  std::vector<int> sorted_tour = best_tour, all_cities(kNumCities);
  std::sort(sorted_tour.begin(), sorted_tour.end());
  for (int i = 0; i < kNumCities; ++i) {
    all_cities[i] = i;
  }
  bool pass = lab::check_equal("tour is a permutation", sorted_tour, all_cities);
  // ... and its reported length must match a recomputation.
  if (pass) {
    double length = 0.0;
    for (int i = 0; i < kNumCities; ++i) {
      length += h_dist[best_tour[i] * kNumCities + best_tour[(i + 1) % kNumCities]];
    }
    const std::vector<float> got = {best_length};
    const std::vector<double> expected = {length};
    pass = lab::check_close("tour length", got, expected, 1e-4, 0.0);
  }

  CUDA_CHECK(cudaFree(d_states));
  CUDA_CHECK(cudaFree(d_tours));
  CUDA_CHECK(cudaFree(d_lengths));
  CUDA_CHECK(cudaFree(d_best_tour));
  CUDA_CHECK(cudaFree(d_best_length));
  return lab::finish(pass);
}
