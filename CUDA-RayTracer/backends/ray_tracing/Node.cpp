#include "Node.h"
#include <cfloat>

Node::Node() = default;

Node::Node(int parent, int myIndex, int *triangles, Box boundingBox, Scene *scene) {
    this->parent = parent;
    this->left = -1;
    this->right = -1;
    this->myIndex = myIndex;
    this->triangles = triangles;
    this->boundingBox = boundingBox;
    this->scene = scene;
}

int Node::getNearestTriangle(Vector &vector, int ignoredIndex) {
    int bestIndex = -1;
    float best = FLT_MAX;

    if (left == -1 && right == -1) {
        for (int i = 0; i < nomNumOfTriangles; ++i) {
			if(this->triangles[i] == ignoredIndex)
				continue;

            float dist = scene->getTriangles()[this->triangles[i]].getDist(vector);

            if (dist != -1 && dist < best) {
                bestIndex = triangles[i];
                best = dist;
            }
        }
    }
    return bestIndex;
}

bool Node::isLeaf() {
    return left == right;
}

Node::~Node() {
    delete[] triangles;
}
