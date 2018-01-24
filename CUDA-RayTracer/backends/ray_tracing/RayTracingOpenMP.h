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
    static constexpr int MAX_DEPTH = 10;
    static constexpr float MINIMUM_WEIGHT = 0.001f;
    static constexpr float FULLY_OPAQUE_RATIO = 0.99f;
    static constexpr float FULLY_TRANSPARENT_RATIO = 0.01f;

    Color *data = nullptr;
    Color Ia;

    std::unique_ptr<KdTree> kdTree;

    Vector refract(const Vector &vector, const Vector &normal, float ior) const;

    float fresnel(const Vector &vector, const Vector &normal, float ior) const;

    Color trace(Vector vector, int depth, int ignoredTriangle = -1, float weight = 1.f);

public:
    RayTracingOpenMP();

    ~RayTracingOpenMP() override;

    Image render() override;

    void setScene(std::unique_ptr<Scene> scene) override;
};

#endif //RAY_TRACER_OPENMP_H
