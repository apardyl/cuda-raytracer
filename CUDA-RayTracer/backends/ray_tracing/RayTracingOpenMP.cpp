#include "RayTracingOpenMP.h"

#include <cmath>

#include "KdTree.h"
#include "Resolution.h"
#include "Camera.h"

const int N = (int) 1e6 + 5;
const float eps = 1e-6;
const int BYTES_PER_PIXEL = 3;

RayTracingOpenMP::RayTracingOpenMP() = default;

RayTracingOpenMP::~RayTracingOpenMP() {
    delete[] data;
}

Image RayTracingOpenMP::render() {
    KdTree kdTree(scene.get());
    Light light(Point(0, 0, -1), Color(1, 1, 1), Color(1, 1, 1));
    kdTree.registerLight(light);
    Resolution resolution = Resolution(width, height);
    Camera camera(Point(0, 0, -1), Point(0, static_cast<float>(M_PI), 0), M_PI_2, resolution, 1);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Vector vector = camera.getPrimaryVector(x, y);
            Color color = kdTree.trace(vector, 20);
            camera.update(x, y, color);
        }
    }
    // return Image
    data = new Color[width * height];
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
