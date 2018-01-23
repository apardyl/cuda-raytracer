#include "KdTree.h"

#include <cmath>
#include <cfloat>

KdTree::KdTree(Scene *scene) {
    this->scene = scene;
    nodes = new Node[1000005];

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

    Node temp(nullptr, getBoundingBox(triangles), scene);
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

KdTree::~KdTree() {
    delete[] nodes;
}
