#ifndef RAY_TRACER_CUDABACKEND_H
#define RAY_TRACER_CUDABACKEND_H

#include "application/CompileSettings.h"
#if CUDA_ENABLED

#include "backends/Backend.h"

/**
 * Base class for the backends utilizing CUDA.
 */
class CudaBackend : public Backend {
protected:
    byte *data = nullptr;

    virtual void doRender() = 0;

public:
    static const unsigned BYTES_PER_PIXEL = 3;

    ~CudaBackend() override;

    Image render() override;

    void setResolution(unsigned width, unsigned height) override;
};

#endif //CUDA_ENABLED
#endif //RAY_TRACER_CUDABACKEND_H
