#ifndef RAY_TRACER_SOLIDCOLORCUDABACKEND_H
#define RAY_TRACER_SOLIDCOLORCUDABACKEND_H

#include <cstddef>
#include "backends/Backend.h"

/**
 * Very simple rendering backend that just fills the surface with a solid color
 * using CUDA.
 */
class SolidColorCudaBackend : public Backend {
private:
    byte *data = nullptr;

public:
    SolidColorCudaBackend();

    ~SolidColorCudaBackend() override;

    Image render() override;

    void setResolution(unsigned width, unsigned height) override;
};

#endif //RAY_TRACER_SOLIDCOLORCUDABACKEND_H
