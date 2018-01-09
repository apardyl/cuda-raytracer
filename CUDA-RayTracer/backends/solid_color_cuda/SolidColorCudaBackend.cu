#include "SolidColorCudaBackend.h"
#if CUDA_ENABLED

#include <cuda_runtime.h>

const int BLOCK_SIZE = 32;
const int BYTES_PER_PIXEL = SolidColorCudaBackend::BYTES_PER_PIXEL;

// Rendering red to make sure the subpixel values are interpreted in the
// correct order
const byte COLOR_RED = 255;
const byte COLOR_GREEN = 0;
const byte COLOR_BLUE = 0;

__global__ void renderSolidColor(byte *data, unsigned width, unsigned height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    data[(width * y + x) * BYTES_PER_PIXEL] = COLOR_RED;
    data[(width * y + x) * BYTES_PER_PIXEL + 1] = COLOR_GREEN;
    data[(width * y + x) * BYTES_PER_PIXEL + 2] = COLOR_BLUE;
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
