/*
 * 3D Stencil 1: Shared-Memory Planes with Register Streaming along z
 *
 * Intention:
 * A block owns a 32 x 8 column of points in x and y and marches through it
 * along z. At each z it needs three planes: below, current, and above.
 * - The current plane (with a one-point halo in x and y) goes into shared
 *   memory, because 5 of the 7 neighbours come from it and other threads
 *   loaded them.
 * - Each thread's own below and above values stay in registers: a thread
 *   needs only its own column from those planes, and when the block steps
 *   from z to z + 1, "current" becomes "below" and "above" becomes "current"
 *   without being re-read.
 * Every input value is now read from global memory about once (plus the
 * halo), instead of up to 7 times.
 *
 * This "register tiling along the streaming dimension" (or "thread
 * coarsening in z") is the standard optimization for structured-grid
 * stencils; see Programming Massively Parallel Processors, chapter 8.
 *
 * High-Level Algorithm, per block:
 *   below = in[z = 0], current = in[z = 1]
 *   for z = 1 .. n - 2:
 *     above = in[z + 1]                (register, one load per thread)
 *     stage current + x/y halo in shared memory; __syncthreads()
 *     out[z] = C0 * current + C1 * (4 shared neighbours + below + above)
 *     below = current; current = above; __syncthreads()
 */
#include "../stencil_harness.cuh"

constexpr int TX = 32;
constexpr int TY = 8;

__global__ void stencil_z_streaming(const float *in, float *out, int n) {
  __shared__ float plane_s[TY + 2][TX + 2];

  const int x = blockIdx.x * TX + threadIdx.x;
  const int y = blockIdx.y * TY + threadIdx.y;
  const int lx = threadIdx.x + 1;  // Position in the shared plane (with halo)
  const int ly = threadIdx.y + 1;
  const bool in_grid = x < n && y < n;
  const bool interior_xy = x >= 1 && x < n - 1 && y >= 1 && y < n - 1;
  const size_t plane = static_cast<size_t>(n) * n;
  const size_t column = static_cast<size_t>(y) * n + x;

  float below = in_grid ? in[column] : 0.0f;
  float current = in_grid ? in[plane + column] : 0.0f;

  for (int z = 1; z < n - 1; ++z) {
    const size_t zoff = z * plane;
    const float above = in_grid ? in[zoff + plane + column] : 0.0f;

    // Stage the current plane and its x/y halo.
    plane_s[ly][lx] = current;
    if (threadIdx.x == 0 && x >= 1 && y < n) {
      plane_s[ly][0] = in[zoff + column - 1];
    }
    if (threadIdx.x == TX - 1 && x + 1 < n && y < n) {
      plane_s[ly][TX + 1] = in[zoff + column + 1];
    }
    if (threadIdx.y == 0 && y >= 1 && x < n) {
      plane_s[0][lx] = in[zoff + column - n];
    }
    if (threadIdx.y == TY - 1 && y + 1 < n && x < n) {
      plane_s[TY + 1][lx] = in[zoff + column + n];
    }
    __syncthreads();

    if (interior_xy) {
      out[zoff + column] =
          C0 * current +
          C1 * (plane_s[ly][lx - 1] + plane_s[ly][lx + 1] +
                plane_s[ly - 1][lx] + plane_s[ly + 1][lx] + below + above);
    }
    below = current;
    current = above;
    __syncthreads();  // Everyone is done with plane_s before it is refilled.
  }
}

void launch(const float *d_in, float *d_out, int n) {
  dim3 block(TX, TY);
  dim3 grid(lab::ceil_div(n, TX), lab::ceil_div(n, TY));
  stencil_z_streaming<<<grid, block>>>(d_in, d_out, n);
}

int main(int argc, char **argv) {
  return run_stencil("1. Shared planes + z streaming", argc, argv, launch);
}
