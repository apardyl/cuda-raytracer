#include "KdTree.h"

#include <cmath>
#include <cfloat>

KdTree::KdTree(Scene *scene) {
    this->scene = scene;
    nodes = new Node[1000005];
    lights = new Light[20];
    Ia = Color(0.2, 0.2, 0.2);

    std::vector<int> triangles;
    for (int i = 0; i < scene->trianglesCount; ++i) {
        triangles.push_back(i);
    }

    buildTree(triangles, -1, 0, 1);
}

int KdTree::getNearestTriangle(Vector &vector, int ignoredIndex) {
    int ans = -1;
    float bestDistance = FLT_MAX;

    Stack stack;
    stack.addElement(0);

    while (stack.size > 0) {
        int cur = stack.top();
        stack.pop();

        if (nodes[cur].boundingBox.isIntersecting(vector)) {
            if (nodes[cur].isLeaf()) {
                int indexOfBestTriangle = nodes[cur].getNearestTriangle(vector, ignoredIndex);

                if (indexOfBestTriangle != -1) {
                    float distance = scene->getTriangles()[indexOfBestTriangle].getDist(vector);

                    if (bestDistance > distance) {
                        bestDistance = distance;
                        ans = indexOfBestTriangle;
                    }
                }
            } else {
                if (nodes[cur].left != -1) {
                    stack.addElement(nodes[cur].left);
                }

                if (nodes[cur].right != -1) {
                    stack.addElement(nodes[cur].right);
                }
            }
        }
    }

    return ans;
}

int KdTree::buildTree(std::vector<int> triangles, int parent, int axis, int depth) {
    int nodeIndex = numberOfNodes++;
    int nextAxis = (axis + 1) % 3;

    Node temp(parent, nodeIndex, nullptr, getBoundingBox(triangles), scene);
    nodes[nodeIndex] = temp;
    Node &cur = nodes[nodeIndex];

    std::vector<int> left, right;

	// if it's leaf
    if (triangles.size() < 10 || depth > 15 || !split(triangles, left, right, axis)) {
        cur.nomNumOfTriangles = triangles.size();
        cur.triangles = new int[triangles.size()];

        for (int i = 0; i < triangles.size(); ++i) {
            cur.triangles[i] = triangles[i];
        }

        return nodeIndex;
    }

    if (!left.empty()) {
        cur.left = buildTree(left, nodeIndex, nextAxis, depth + 1);
    }

    if (!right.empty()) {
        cur.right = buildTree(right, nodeIndex, nextAxis, depth + 1);
    }

    return nodeIndex;
}

const Color BACKGROUND_COLOR(0, 0, 0);

Color KdTree::trace(Vector vector, int depth, int ignoredTriangle) {
    if (depth > 20) {
        return BACKGROUND_COLOR;
    }

    int triangleIndex = getNearestTriangle(vector, ignoredTriangle);
    if (triangleIndex == -1) {
        return BACKGROUND_COLOR;
    }

    Triangle &triangle = scene->getTriangles()[triangleIndex];
    Vector reflectionVector = triangle.getReflectedVector(vector);
    reflectionVector.normalize();

    Point reflectionPoint = reflectionVector.startPoint;

    Vector normal = scene->getTriangles()[triangleIndex].getNormal();

    if (normal.isObtuse(reflectionVector)) {
        normal = normal.mul(-1);
    }

    Material material = scene->getMaterial((scene->getTriangles()[triangleIndex]).materialCode);

    Vector toViewer = vector.mul(-1);
    Color refractionColor(0, 0, 0);
    Color reflectionColor = Ia * material.ambient;
    float refractivity = 0;

    if (material.dissolve > 0.01f) {
        for (int light = 0; light < numberOfLights; ++light) {
            Vector toLight = Vector(reflectionPoint, lights[light].point);
            toLight.normalize();

            // Check if the light is blocked out
            if (normal.isObtuse(toLight)) {
                continue;
            }

            // Cast shadow ray, take refraction into account (simplified version)
            float intensity = 1;
            float dist = 0;
            float lightDistance = lights[light].point.getDist(reflectionPoint);
            int lightTriangleIndex = triangleIndex;
            while (intensity > 0.01f) {
                lightTriangleIndex = getNearestTriangle(toLight, lightTriangleIndex);

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
            Vector fromLight(lights[light].point, reflectionPoint);
            Vector fromLightReflected = scene->getTriangles()[triangleIndex].getReflectedVector(
                    fromLight);
            fromLightReflected.normalize();

            reflectionColor +=
                    lights[light].Id * intensity *
                    std::max(0.f, normal.dot(toLight)) * material.diffuse;
            reflectionColor +=
                    lights[light].Is * intensity *
                    powf(std::max(0.f, toViewer.dot(fromLightReflected)),
                         material.specularExponent) *
                    material.specular;
        }

        reflectionColor +=
                trace(reflectionVector, depth + 1, triangleIndex) *
                powf(std::max(0.f, toViewer.dot(normal)), material.specularExponent) *
                material.specular;
    }

    if (material.dissolve < 0.99f) {
        // todo get ior from the material
        refractionColor = trace(
                refract(vector, triangle.getNormal(), 1.5), depth + 1, triangleIndex);
        float reflectivity = fresnel(vector, normal, 1.5);
        refractivity = (1 - reflectivity) * (1 - material.dissolve);
    }

    if (material.dissolve >= 0.99f) {
        return reflectionColor;
    } else if (material.dissolve <= 0.01f) {
        return refractionColor;
    }

    return reflectionColor * (1 - refractivity) + refractionColor * refractivity;
}

Vector KdTree::refract(const Vector &vector, const Vector &normal, float ior) const {
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
    float k = (1 - eta * eta) * (1 - dot * dot);
    if (k < 0) {
        // Total internal reflection
        return Vector::ZERO;
    }
    return vector.mul(dot).add(localNormal.mul(eta * dot - sqrtf(k)));
}

float KdTree::fresnel(const Vector &vector, const Vector &normal, float ior) const {
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

Box KdTree::getBoundingBox(std::vector<int> &triangles_) {
    Point minPoint(FLT_MAX, FLT_MAX, FLT_MAX);

	// width, length, height
    float x = 0, y = 0, z = 0;

    for (auto &index : triangles_) {
        Triangle &triangle = scene->getTriangles()[index];
        minPoint.x = std::min(minPoint.x, triangle.getMinX());
        minPoint.y = std::min(minPoint.y, triangle.getMinY());
        minPoint.z = std::min(minPoint.z, triangle.getMinZ());
    }

    Point maxPoint(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    for (auto &index : triangles_) {
        Triangle &triangle = scene->getTriangles()[index];
        maxPoint.x = std::max(maxPoint.x, triangle.getMaxX());
        maxPoint.y = std::max(maxPoint.y, triangle.getMaxY());
        maxPoint.z = std::max(maxPoint.z, triangle.getMaxZ());
    }

    x = maxPoint.x - minPoint.x;
    y = maxPoint.y - minPoint.y;
    z = maxPoint.z - minPoint.z;

    return Box(minPoint, x, y, z);
}

bool KdTree::split(std::vector<int> &triangles,
                   std::vector<int> &left,
                   std::vector<int> &right,
                   int axis) {
    if (triangles.size() <= 1) {
        return false;
    }

    std::function<bool(const int &a, const int &b)> com = nullptr;
    switch (axis) {
        case 0:
            com = comByX;
            break;
        case 1:
            com = comByY;
            break;
        case 2:
            com = comByZ;
            break;
    }

    std::sort(triangles.begin(), triangles.end(), com);
    int midTriangle = triangles[triangles.size() / 2];

    for (auto &triangle : triangles) {
        if (com(triangle, midTriangle)) {
            left.push_back(triangle);
        } else {
            right.push_back(triangle);
        }
    }

    return true;
}

void KdTree::registerLight(Light light) {
    lights[numberOfLights++] = light;
}

KdTree::~KdTree() {
    delete[] nodes;
    delete[] lights;
}
