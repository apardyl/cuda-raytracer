#ifndef RAY_TRACER_OPENMP_H
#define RAY_TRACER_OPENMP_H

#include "../Backend.h"
#include "KdTree.h"

/**
 * Very simple rendering backend that just fills the surface with a solid color
 * using CUDA.
 */
class RayTracingOpenMP : public Backend {
private:
    Color *data = nullptr;
    Light *lights = nullptr;
    int numberOfLights = 0;
    Color Ia;

    std::unique_ptr<KdTree> kdTree;

    Vector refract(const Vector &vector, const Vector &normal, float ior) const;

    float fresnel(const Vector &vector, const Vector &normal, float ior) const;

    Color trace(Vector vector, int depth, int ignoredTriangle = -1);
public:
    RayTracingOpenMP();

    ~RayTracingOpenMP() override;

    Image render() override;

    void setScene(std::unique_ptr<Scene> scene) override;
};

#endif //RAY_TRACER_OPENMP_H
