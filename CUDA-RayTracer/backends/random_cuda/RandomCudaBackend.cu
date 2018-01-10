#include "RandomCudaBackend.h"
#if CUDA_ENABLED

#include "scene/Color.h"
#include <cuda_runtime.h>
#include <ctime>
#include <climits>

const int BLOCK_SIZE = 32;

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

__global__ void renderRandomColor(Color * data, unsigned width, unsigned height,
                                  unsigned seed, unsigned long long *states) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width || y >= height) {
        return;
    }

    unsigned long long *state = &states[y * width + x];
    randInit(state, x, y, seed);

    data[(width * y + x)].red = static_cast<double>(xorshift64star(state)) / UINT_MAX;
    data[(width * y + x)].green = static_cast<double>(xorshift64star(state)) / UINT_MAX;
    data[(width * y + x)].blue = static_cast<double>(xorshift64star(state)) / UINT_MAX;
}

void RandomCudaBackend::doRender() {
    const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
    const dim3 gridSize(
        width + (BLOCK_SIZE - 1) / BLOCK_SIZE,
        height + (BLOCK_SIZE - 1) / BLOCK_SIZE,
        1);

    unsigned long long *states;
    cudaMalloc((void**) &states, width * height * sizeof(unsigned long long));

    renderRandomColor<<<gridSize, blockSize>>>(
        data, width, height, time(0), states);

    cudaDeviceSynchronize();
    cudaFree(states);
}

#endif //CUDA_ENABLED
