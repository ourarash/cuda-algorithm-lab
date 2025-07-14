#include <cstdio>

__global__ void shfl_example_kernel() {
    int laneId = threadIdx.x % 32;
    int value = laneId;

    int bcast = __shfl_sync(0xFFFFFFFF, value, 0);      // Broadcast from lane 0
    int up    = __shfl_up_sync(0xFFFFFFFF, value, 1);    // Value from laneId - 1
    int xorv  = __shfl_xor_sync(0xFFFFFFFF, value, 1);   // Value from laneId ^ 1

    printf("Thread %d: val=%d, bcast=%d, up=%d, xor=%d\n", threadIdx.x, value, bcast, up, xorv);
}

int main() {
    shfl_example_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}
