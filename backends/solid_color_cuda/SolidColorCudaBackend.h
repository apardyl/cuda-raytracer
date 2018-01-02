#ifndef RAY_TRACER_RANDOMCUDABACKEND_H
#define RAY_TRACER_RANDOMCUDABACKEND_H

#include <cstddef>
#include "backends/Backend.h"

typedef unsigned char byte;

/**
 * Very simple renderick backend that just fills the surface with a solid color
 * using CUDA.
 */
class SolidColorCudaBackend : public Backend {
private:
    byte *data = nullptr;

public:
    SolidColorCudaBackend();

    ~SolidColorCudaBackend();

    byte *render() override;

    void setResolution(unsigned width, unsigned height) override;
};

#endif //RAY_TRACER_RANDOMCUDABACKEND_H
