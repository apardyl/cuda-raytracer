#include "RayTracingOpenMP.h"

#include <boost/math/constants/constants.hpp>

#include "Camera.h"
#include "KdTree.h"

namespace math = boost::math::constants;

const Color BACKGROUND_COLOR(0, 0, 0);

RayTracingOpenMP::RayTracingOpenMP() {
    Ia = Color(0.2, 0.2, 0.2);
}

RayTracingOpenMP::~RayTracingOpenMP() {
    delete[] data;
}

Image RayTracingOpenMP::render() {
    Resolution resolution = Resolution(width, height);
    Camera camera(scene->camera, resolution, 1);
#pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Vector vector = camera.getPrimaryVector(x, y);
            Color color = trace(vector, 0);
            camera.update(x, y, color);
        }
    }

    data = new Color[width * height];
#pragma omp parallel for
    for (int y = 0; y < resolution.height; ++y) {
        for (int x = 0; x < resolution.width; ++x) {
            data[width * (height - y - 1) + x] = camera.getPixelColor(x, y);
        }
    }
    return Image(resolution.width, resolution.height, data);
}

Color RayTracingOpenMP::trace(Vector vector, int depth, int ignoredTriangle, float weight) {
    if (depth > MAX_DEPTH || weight < MINIMUM_WEIGHT) {
        return BACKGROUND_COLOR;
    }

    int triangleIndex = kdTree->getNearestTriangle(vector, ignoredTriangle);
    if (triangleIndex == -1) {
        return BACKGROUND_COLOR;
    }

    Triangle &triangle = scene->getTriangles()[triangleIndex];
    Vector reflectionVector = triangle.getReflectedVector(vector);
    reflectionVector.normalize();

    Point reflectionPoint = reflectionVector.startPoint;

    Vector normal = scene->getTriangles()[triangleIndex].getNormal();

    Material material = scene->getMaterial((scene->getTriangles()[triangleIndex]).materialCode);

    Vector toViewer = vector.mul(-1);
    Color refractionColor(0, 0, 0);
    Color reflectionColor = Ia * material.ambient;
    float refractivity = 0;

    if (material.dissolve < FULLY_OPAQUE_RATIO) {
        float ior = material.refractiveIndex;
        float reflectivity = fresnel(vector, normal, ior);
        refractivity = (1 - reflectivity) * (1 - material.dissolve);
        Vector refractionVector = refract(vector, triangle.getNormal(), ior);

        refractionVector.startPoint = reflectionPoint;
        refractionColor =
            trace(refractionVector, depth + 1, triangleIndex, weight * refractivity) *
            material.transparent;
    }

    if (material.dissolve > FULLY_TRANSPARENT_RATIO) {
        for (int light = 0; light < scene->lightsCount; ++light) {
            Vector toLight = Vector(reflectionPoint, scene->getLights()[light].point);
            toLight.normalize();

            // Check if the light is blocked out
            if (normal.isObtuse(toLight)) {
                continue;
            }

            // Cast shadow ray, take refraction into account (simplified version)
            float intensity = 1;
            float dist = 0;
            float lightDistance = scene->getLights()[light].point.getDist(reflectionPoint);
            int lightTriangleIndex = triangleIndex;
            for (int lightDepth = depth;
                 lightDepth < MAX_DEPTH && intensity > 0.01f; ++lightDepth) {
                lightTriangleIndex = kdTree->getNearestTriangle(toLight, lightTriangleIndex);

                if (lightTriangleIndex == -1) {
                    break;
                }
                const Intersection &intersection =
                    scene->getTriangles()[lightTriangleIndex].intersect(toLight);
                dist += intersection.distance;
                if (dist >= lightDistance) {
                    break;
                }

                toLight.startPoint = intersection.point;
                intensity *= (1 - scene->getMaterial(triangle.materialCode).dissolve);
            }
            if (intensity <= 0.01f) {
                continue;
            }

            // Calculate reflection color
            Vector fromLight(scene->getLights()[light].point, reflectionPoint);
            Vector fromLightReflected = scene->getTriangles()[triangleIndex].getReflectedVector(
                fromLight);
            fromLightReflected.normalize();

            reflectionColor +=
                scene->getLights()[light].diffuse * intensity *
                std::max(0.f, normal.dot(toLight)) * material.diffuse;
            reflectionColor +=
                scene->getLights()[light].specular * intensity *
                powf(std::max(0.f, toViewer.dot(fromLightReflected)),
                     material.specularExponent) * material.specular;
        }

        reflectionColor +=
            trace(reflectionVector, depth + 1, triangleIndex, weight * (1 - refractivity)) *
            powf(std::max(0.f, toViewer.dot(normal)), material.specularExponent) *
            material.specular;
    }

    if (material.dissolve >= FULLY_OPAQUE_RATIO) {
        return reflectionColor;
    } else if (material.dissolve <= FULLY_TRANSPARENT_RATIO) {
        return refractionColor;
    }

    return reflectionColor * (1 - refractivity) + refractionColor * refractivity;
}

Vector RayTracingOpenMP::refract(const Vector &vector, const Vector &normal, float ior) const {
    float dot = vector.dot(normal);
    float eta1 = 1;
    float eta2 = ior;
    Vector localNormal = normal;

    if (dot < 0) {
        // Ray entering the object
        dot *= -1;
    } else {
        // Ray going out of the object
        localNormal = normal.mul(-1);
        std::swap(eta1, eta2);
    }

    float eta = eta1 / eta2;
    float k = 1 - eta * eta * (1 - dot * dot);
    if (k < 0) {
        // Total internal reflection
        return Vector::ZERO;
    }

    Vector returnVector = vector.mul(eta).add(localNormal.mul(eta * dot - sqrtf(k)));
    returnVector.normalize();
    return returnVector;
}

float RayTracingOpenMP::fresnel(const Vector &vector, const Vector &normal, float ior) const {
    float cos1 = vector.dot(normal);
    float eta1 = 1;
    float eta2 = ior;

    if (cos1 > 0) {
        std::swap(eta1, eta2);
    }

    float sin2 = eta1 / eta2 * sqrtf(std::max(0.f, 1 - cos1 * cos1));
    if (sin2 >= 1.f) {
        // Total internal reflection
        return 1;
    }

    float cos2 = sqrtf(1 - sin2 * sin2);
    cos1 = fabsf(cos1);
    float reflectS = (eta1 * cos1 - eta2 * cos2) / (eta1 * cos1 + eta2 * cos2);
    float reflectP = (eta1 * cos2 - eta2 * cos1) / (eta1 * cos2 + eta2 * cos1);
    return (reflectS * reflectS + reflectP * reflectP) / 2;
}

void RayTracingOpenMP::setScene(std::unique_ptr<Scene> scene) {
    Backend::setScene(std::move(scene));
    kdTree = std::make_unique<KdTree>(this->scene.get());
}
