﻿#ifndef RAY_TRACER_NODE_H
#define RAY_TRACER_NODE_H

#include "Box.h"
#include "scene/Scene.h"

struct Node {
    int parent;
    int left;
    int right;
    int myIndex;
    int *triangles = nullptr;
    size_t nomNumOfTriangles = 0;

    Box boundingBox;
    Scene *scene = nullptr;

    Node();

    Node(int parent, int my_index, int *triangles, Box bounding_box, Scene *scene);

    // If there is no such triangle return -1
    int getNearestTriangle(Vector &vector, int ignoredIndex);

    bool isLeaf();

    ~Node();
};

#endif //RAY_TRACER_NODE_H
