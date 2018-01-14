#include "Camera.h"

#include <cmath>

#include "scene/Vector.h"

Camera::Camera(Point location, Point rotation, float horizontalFOV, const Resolution &resolution,
               int numOfSamples) :
        location(location),
        rotation(rotation),
        horizontalFOV(horizontalFOV),
        width(2 * std::sin(horizontalFOV / 2)),
        height(width * resolution.height / resolution.width),
        resolution(resolution),
        numOfSamples(numOfSamples),
        activePixelSensor(std::make_unique<Color[]>(resolution.width * resolution.height)) {
    for (int y = 0; y < resolution.height; ++y) {
        for (int x = 0; x < resolution.width; ++x) {
            getActivePixelSensor(x, y) = Color(0, 0, 0);
        }
    }
}

Color &Camera::getActivePixelSensor(int x, int y) const {
    return activePixelSensor[resolution.width * y + x];
}

Vector Camera::getRandomVector(int x, int y) const { // not implemented
    return Vector(Point(0, 0, 0), 0, 0, 0);
}

Vector Camera::getPrimaryVector(int x, int y) const {
    float pixelWidth = width / resolution.width;
    float pixelHeight = height / resolution.height;
    float xCoord = (-width / 2) + pixelWidth * (0.5 + x);
    float yCoord = (-height / 2) + pixelHeight * (0.5 + y);

    Vector vector = Vector(Point(0, 0, 0), Point(yCoord, xCoord, -1))
            .rotateX(rotation.x)
            .rotateY(rotation.y)
            .rotateZ(rotation.z);
    vector.startPoint = location;
    return vector.normalize();
}

void Camera::update(int x, int y, const Color &color) {
    getActivePixelSensor(x, y) += color;
}

Color Camera::getPixelColor(int x, int y) const {
    return getActivePixelSensor(x, y) / (float) numOfSamples;
}
