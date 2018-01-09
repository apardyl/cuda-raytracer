#ifndef RAY_TRACER_RANDOMCUDABACKEND_H
#define RAY_TRACER_RANDOMCUDABACKEND_H

#if CUDA_ENABLED

#include "backends/cuda_backend/CudaBackend.h"

/**
 * Simple rendering backend that fills the surface with random pixels using
 * CUDA.
 */
class RandomCudaBackend : public CudaBackend {
protected:
    void doRender() override;
};

#endif //CUDA_ENABLED
#endif //RAY_TRACER_RANDOMCUDABACKEND_H
