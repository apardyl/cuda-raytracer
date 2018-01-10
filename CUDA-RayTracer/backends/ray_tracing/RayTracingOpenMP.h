#ifndef RAY_TRACER_OPENMP_H
#define RAY_TRACER_OPENMP_H

#include "../Backend.h"

/**
 * Very simple rendering backend that just fills the surface with a solid color
 * using CUDA.
 */
class RayTracingOpenMP : public Backend {
private:
    byte *data = nullptr;

public:
    RayTracingOpenMP();

    ~RayTracingOpenMP() override;

    Image render() override;

    void setSoftShadows(bool var);
};

#endif //RAY_TRACER_OPENMP_H
