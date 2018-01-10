#include "SolidColorCudaBackend.h"
#if CUDA_ENABLED
#include "scene/Color.h"
#include <cuda_runtime.h>

const int BLOCK_SIZE = 32;

__global__ void renderSolidColor(Color *data, unsigned width, unsigned height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    data[(width * y + x)].red = 1;
    data[(width * y + x)].green = 0;
    data[(width * y + x)].blue = 0;
}

void SolidColorCudaBackend::doRender() {
    const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
    const dim3 gridSize(
            width + (BLOCK_SIZE - 1) / BLOCK_SIZE,
            height + (BLOCK_SIZE - 1) / BLOCK_SIZE,
            1);
    renderSolidColor<<<gridSize, blockSize>>>(data, width, height);
}

#endif
