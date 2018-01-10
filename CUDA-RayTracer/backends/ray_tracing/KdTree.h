#pragma once

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
    int numberOfLights = 0;
    Light *lights = nullptr;
    Color Ia;

    explicit KdTree(Scene *scene);

    int get_triangle(Vector &vector);
    // get triangle which have collison with vector // if there isn't any triangle return -1

    int build_tree(std::vector<int> triangles, int parent, int axis, int depth);

    Color trace(Vector vector, int depth);

    Box get_bounding_box(std::vector<int> &triangles_);

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
