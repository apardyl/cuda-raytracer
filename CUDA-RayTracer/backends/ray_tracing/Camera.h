#ifndef RAY_TRACER_CAMERA_H
#define RAY_TRACER_CAMERA_H

#include "Camera.h"
#include "Resolution.h"
#include "scene/Point.h"
#include "scene/Color.h"

#include <memory>

class Camera {
private:
    Point location;
    Point rotation;
    float horizontalFOV;

    float width, height;
    Resolution resolution;
    int numOfSamples = 1024;
    std::unique_ptr<Color[]> activePixelSensor;

    Color &getActivePixelSensor(int x, int y) const;

public:
    /**
     * @param location location of the camera
     * @param rotation rotation of the camera in radians. (0, 0, 0) rotation means bottom
     *  (negative Z axis).
     * @param horizontalFOV horizontal field of view (FOV), in radians
     * @param resolution resolution of the camera, in pixels
     * @param numOfSamples number of samples computed for each pixel
     */
    Camera(Point location, Point rotation, float horizontalFOV, const Resolution &resolution,
           int numOfSamples);

    Vector getRandomVector(int x, int y) const;

    Vector getPrimaryVector(int x, int y) const;

    void update(int x, int y, const Color &color);

    Color getPixelColor(int x, int y) const;
};

#endif //RAY_TRACER_CAMERA_H
