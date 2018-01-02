#include "SolidColorCudaBackend.h"

const int BLOCK_SIZE = 32;
const int BYTES_PER_PIXEL = 3;

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

SolidColorCudaBackend::~SolidColorCudaBackend() {
    cudaFree(data);
}

byte *SolidColorCudaBackend::render() {
    const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
    const dim3 gridSize(
            width + (BLOCK_SIZE - 1) / BLOCK_SIZE,
            height + (BLOCK_SIZE - 1) / BLOCK_SIZE,
            1);
    renderSolidColor<<<gridSize, blockSize>>>(data, width, height);
    cudaDeviceSynchronize();
    return data;
}

void SolidColorCudaBackend::setResolution(unsigned width, unsigned height) {
    Backend::setResolution(width, height);

    cudaFree(data);
    cudaMallocHost(&data, sizeof(byte) * width * height * BYTES_PER_PIXEL);
}
