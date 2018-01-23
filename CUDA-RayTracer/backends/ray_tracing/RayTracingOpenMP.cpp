#include "RayTracingOpenMP.h"
#include "Camera.h"
#include "Resolution.h"
#include "KdTree.h"
#include <cmath>
#include <boost/math/constants/constants.hpp>

namespace math = boost::math::constants;

RayTracingOpenMP::RayTracingOpenMP() = default;

RayTracingOpenMP::~RayTracingOpenMP() {
    delete[] data;
}

Image RayTracingOpenMP::render() {
    KdTree kdTree(scene.get());
    for (int i = 0; i < scene->lightsCount; i++) {
        kdTree.registerLight(scene->getLights()[i]);
    }
    Resolution resolution = Resolution(width, height);
    Camera camera(Point(0, 0, -1), Point(0, math::pi<float>(), 0),
                  math::pi<float>() / 2, resolution, 1);
#pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Vector vector = camera.getPrimaryVector(x, y);
            Color color = kdTree.trace(vector, 20);
            camera.update(x, y, color);
        }
    }
    // return Image
    data = new Color[width * height];
#pragma omp parallel for
    for (int y = 0; y < resolution.height; ++y) {
        for (int x = 0; x < resolution.width; ++x) {
            data[width * y + x] = camera.getPixelColor(x, y);
        }
    }
    return Image(resolution.width, resolution.height, data);
}

void RayTracingOpenMP::setSoftShadows(bool var) {
    // not implemented
}
