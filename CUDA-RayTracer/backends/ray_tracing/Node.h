#pragma once

#include "scene/Scene.h"
#include "Box.h"

struct Node {
    int parent, left, right, my_index;
    int *triangles = nullptr;
    int num_of_triangles = 0;
    Box bounding_box;
    Scene *scene = nullptr;

    Node();

    Node(int parent, int my_index, int *triangles, Box bounding_box, Scene *scene);

    int get_minimal_triangle(Vector &vector); // if there is no such triangle return -1

    bool is_leaf();

    ~Node();
};
