#include "RandomCudaBackend.h"

#include <cuda_runtime.h>
#include <ctime>

const int BLOCK_SIZE = 32;
const int BYTES_PER_PIXEL = RandomCudaBackend::BYTES_PER_PIXEL;

__device__ unsigned xorshift64star(unsigned long long *state) {
    unsigned long long x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return (x * 0x2545F4914F6CDD1D) >> 32;
}

__device__ void randInit(unsigned long long *state, int x, int y, int seed) {
    *state = (x * 10000000019 + y * 100003) ^ (seed << 16) + 1;
    int i1 = (*state & 0x1F);
    for(int i = 0; i < i1; i++) {
        xorshift64star(state);
    }
}

__global__ void renderRandomColor(byte *data, unsigned width, unsigned height,
                                  unsigned seed, unsigned long long *states) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width || y >= height) {
        return;
    }

    unsigned long long *state = &states[y * width + x];
    randInit(state, x, y, seed);

    data[(width * y + x) * BYTES_PER_PIXEL] = xorshift64star(state) % 255;
    data[(width * y + x) * BYTES_PER_PIXEL + 1] = xorshift64star(state) % 255;
    data[(width * y + x) * BYTES_PER_PIXEL + 2] = xorshift64star(state) % 255;
}

void RandomCudaBackend::doRender() {
    const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
    const dim3 gridSize(
        width + (BLOCK_SIZE - 1) / BLOCK_SIZE,
        height + (BLOCK_SIZE - 1) / BLOCK_SIZE,
        1);

    unsigned long long *states;
    cudaMalloc((void**) &states, width * height * sizeof(unsigned long long));

    renderRandomColor << <gridSize, blockSize >> >(
        data, width, height, time(0), states);

    cudaDeviceSynchronize();
    cudaFree(states);
}
