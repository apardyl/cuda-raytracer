#include "Node.h"
#include <cfloat>

Node::Node() = default;

Node::Node(int parent, int my_index, int *triangles, Box bounding_box, Scene *scene) {
    this->parent = parent;
    this->left = -1;
    this->right = -1;
    this->my_index = my_index;
    this->triangles = triangles;
    this->bounding_box = bounding_box;
    this->scene = scene;
}

int Node::get_minimal_triangle(Vector &vector) {
    int best_index = -1;
    float best = FLT_MAX;
    if (left == -1 && right == -1) {
        for (int i = 0; i < num_of_triangles; ++i) {
            float dist = scene->getTriangles()[this->triangles[i]].getDist(vector);
            if (dist != -1 && dist < best) {
                best_index = triangles[i];
                best = dist;
            }
        }
    }
    return best_index;
}

bool Node::is_leaf() {
    return left == right;
}

Node::~Node() {
    delete[] triangles;
}
