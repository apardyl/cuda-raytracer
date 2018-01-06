#ifndef RAY_TRACER_RANDOMCUDABACKEND_H
#define RAY_TRACER_RANDOMCUDABACKEND_H

#include "backends/cuda_backend/CudaBackend.h"

/**
 * Simple rendering backend that fills the surface with random pixels using
 * CUDA.
 */
class RandomCudaBackend : public CudaBackend {
protected:
    void doRender() override;
};

#endif //RAY_TRACER_RANDOMCUDABACKEND_H
