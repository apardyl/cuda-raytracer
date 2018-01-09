#ifndef RAY_TRACER_SOLIDCOLORCUDABACKEND_H
#define RAY_TRACER_SOLIDCOLORCUDABACKEND_H

#include "backends/cuda_backend/CudaBackend.h"

#if CUDA_ENABLED

/**
 * Very simple rendering backend that just fills the surface with a solid color
 * using CUDA.
 */
class SolidColorCudaBackend : public CudaBackend {
protected:
    void doRender() override;
};

#endif //CUDA_ENABLED
#endif //RAY_TRACER_SOLIDCOLORCUDABACKEND_H
