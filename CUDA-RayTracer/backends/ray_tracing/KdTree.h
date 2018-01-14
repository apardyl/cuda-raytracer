#ifndef RAY_TRACER_KDTREE_H
#define RAY_TRACER_KDTREE_H

#include <iostream>
#include <vector>
#include <functional>
#include "Box.h"
#include "Node.h"
#include "Stack.h"
#include "scene/Scene.h"
#include "scene/Light.h"

struct KdTree {
    Scene *scene = nullptr;

    Node *nodes = nullptr;
	int numberOfNodes = 0;

	Light *lights = nullptr;
	int numberOfLights = 0;
    
    Color Ia;

    explicit KdTree(Scene *scene);

	// get triangle which have collison with vector // if there isn't any triangle return -1
    int getNearestTriangle(Vector &vector, int ingnoredIndex);

    int buildTree(std::vector<int> triangles, int parent, int axis, int depth);

    Vector refract(const Vector &vector, const Vector &normal, float ior) const;

    float fresnel(const Vector &vector, const Vector &normal, float ior) const;

    Color trace(Vector vector, int depth, int ignoredTriangle = -1);

    Box getBoundingBox(std::vector<int> &triangles);

    /// Comparators

    std::function<bool(const int &a, const int &b)> comByX = [this](const int &a, const int &b) {
        return this->scene->getTriangles()[a].getMidpoint().x <
               this->scene->getTriangles()[b].getMidpoint().x;
    };

    std::function<bool(const int &a, const int &b)> comByY = [this](const int &a, const int &b) {
        return this->scene->getTriangles()[a].getMidpoint().y <
               this->scene->getTriangles()[b].getMidpoint().y;
    };

    std::function<bool(const int &a, const int &b)> comByZ = [this](const int &a, const int &b) {
        return this->scene->getTriangles()[a].getMidpoint().z <
               this->scene->getTriangles()[b].getMidpoint().z;
    };

    ///

    bool split(std::vector<int> &triangles, std::vector<int> &left, std::vector<int> &right,
               int axis);

    void registerLight(Light light);

    ~KdTree();
};
#endif //RAY_TRACER_KDTREE_H
