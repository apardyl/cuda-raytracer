#include "RandomCudaBackend.h"

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <ctime>

const int BLOCK_SIZE = 32;
const int BYTES_PER_PIXEL = RandomCudaBackend::BYTES_PER_PIXEL;

__global__ void renderSolidColor(byte *data, unsigned width, unsigned height,
                                 unsigned seed, curandState_t* states) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    curandState_t *state = &states[y * width + x];
    curand_init(seed, (width * y + x), 0, state);

    data[(width * y + x) * BYTES_PER_PIXEL] = curand(state) % 255;
    data[(width * y + x) * BYTES_PER_PIXEL + 1] = curand(state) % 255;
    data[(width * y + x) * BYTES_PER_PIXEL + 2] = curand(state) % 255;
}

void RandomCudaBackend::doRender() {
    const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
    const dim3 gridSize(
            width + (BLOCK_SIZE - 1) / BLOCK_SIZE,
            height + (BLOCK_SIZE - 1) / BLOCK_SIZE,
            1);

    curandState_t* states;
    cudaMalloc((void**) &states, width * height * sizeof(curandState_t));

    renderSolidColor<<<gridSize, blockSize>>>(
            data, width, height, time(0), states);

    cudaDeviceSynchronize();
    cudaFree(states);
}
