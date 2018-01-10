#include "RayTracingOpenMP.h"
#include <cmath>
#include "KdTree.h"
#include <SDL_stdinc.h>

const int N = (int) 1e6 + 5;
const float eps = 1e-6;
const int BYTES_PER_PIXEL = 3;


struct Resolution {
    int width, height;

    Resolution() = default;

    Resolution(int width, int height) {
        this->width = width;
        this->height = height;
    }
};

struct Random {
};

struct Camera {
    Point location;
    Point rotation;

    float width, height;
    Resolution resolution;
    int num_of_samples = 1024;
    std::unique_ptr<Color[]> active_pixel_sensor;
    Random random;

    Camera(Point location, Point rotation, float width, float height, Resolution resolution,
           int num_of_samples) :
            location(location),
            rotation(rotation),
            width(width),
            height(height),
            resolution(resolution),
            num_of_samples(num_of_samples),
            active_pixel_sensor(std::make_unique<Color[]>(resolution.width * resolution.height)) {
        for (int y = 0; y < resolution.height; ++y) {
            for (int x = 0; x < resolution.width; ++x) {
                getActivePixelSensor(x, y) = Color(0, 0, 0);
            }
        }
    }

    Color &getActivePixelSensor(int x, int y) const {
        return active_pixel_sensor[resolution.width * y + x];
    }

    Vector get_random_vector(int x, int y) const { // not implemented
        return Vector(Point(0, 0, 0), 0, 0, 0);
    }

    Vector get_primary_vector(int x, int y) const {
        float pixel_width = width / resolution.width;
        float pixel_height = height / resolution.height;
        float x_cord = (-width / 2) + pixel_width * (0.5 + x);
        float y_cord = (-height / 2) + pixel_height * (0.5 + y);
        Vector vector = Vector(Point(0, 0, 0), Point(y_cord, x_cord, -1))
                .rotateX(rotation.x)
                .rotateY(rotation.y)
                .rotateZ(rotation.z);
        vector.startPoint = location;
        return vector;
    }

    void update(int x, int y, Color color) {
        getActivePixelSensor(x, y) += color;
    }

    Color get_pixel_color(int x, int y) const {
        return getActivePixelSensor(x, y) / (float) num_of_samples;
    }
};

RayTracingOpenMP::RayTracingOpenMP() = default;

RayTracingOpenMP::~RayTracingOpenMP() {
    delete[] data;
}

Image RayTracingOpenMP::render() {
    KdTree kdTree(scene.get());
    Light light(Point(0, 0, -1), Color(255, 255, 255), Color(255, 255, 255));
    kdTree.registerLight(light);
    Resolution resolution = Resolution(width, height);
    Camera camera(Point(0, 0, -1), Point(0, static_cast<float>(M_PI), 0), 2, 2, resolution, 1);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Vector vector = camera.get_primary_vector(x, y);
            Color color = kdTree.trace(vector, 20);
            camera.update(x, y, color);
        }
    }
    // return Image
    data = new byte[width * height * BYTES_PER_PIXEL];
    for (int y = 0; y < resolution.height; ++y) {
        for (int x = 0; x < resolution.width; ++x) {
            Color color = camera.get_pixel_color(x, y);
            data[(width * y + x) * BYTES_PER_PIXEL] = std::min(color.red, (float) 255);
            data[(width * y + x) * BYTES_PER_PIXEL + 1] = std::min(color.green, (float) 255);
            data[(width * y + x) * BYTES_PER_PIXEL + 2] = std::min(color.blue, (float) 255);
        }
    }
    return Image(resolution.width, resolution.height, data);
}

void RayTracingOpenMP::setSoftShadows(bool var) {
    // not implemented
}
