#include "CudaBackend.h"
#if CUDA_ENABLED

#include <cuda_runtime.h>

CudaBackend::~CudaBackend() {
    cudaFree(data);
}

Image CudaBackend::render() {
    doRender();
    cudaDeviceSynchronize();
    return Image(width, height, data);
}

void CudaBackend::setResolution(unsigned width, unsigned height) {
    Backend::setResolution(width, height);
    cudaFree(data);
    cudaMallocHost(&data, sizeof(Color) * width * height);
}

#endif //CUDA_ENABLED
