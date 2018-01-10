#include "KdTree.h"
#include "Node.h"

#include <cmath>
#include <cfloat>

KdTree::KdTree(Scene *scene) {
    this->scene = scene;
    nodes = new Node[1000005];
    lights = new Light[20];
    Ia = Color(0.2, 0.2, 0.2);
    std::vector<int> triangles;
    for (int i = 0; i < scene->trianglesCount; ++i)
        triangles.push_back(i);
    build_tree(triangles, -1, 0, 1);
}

int KdTree::get_triangle(Vector &vector) {
    int ans = -1;
    float best_distance = FLT_MAX;
    Stack stack;
    stack.add_element(0);
    while (stack.size > 0) {
        int cur = stack.top();
        stack.pop();
        if (nodes[cur].bounding_box.is_intersecting(vector)) {
            if (nodes[cur].is_leaf()) {
                int index_of_best_triangle = nodes[cur].get_minimal_triangle(vector);
                if (index_of_best_triangle != -1) {
                    float distance = scene->getTriangles()[index_of_best_triangle].getDist(vector);
                    if (best_distance > distance) {
                        best_distance = distance;
                        ans = index_of_best_triangle;
                    }
                }
            } else {
                if (nodes[cur].left != -1)
                    stack.add_element(nodes[cur].left);
                if (nodes[cur].right != -1)
                    stack.add_element(nodes[cur].right);
            }
        }
    }
    return ans;
}

int KdTree::build_tree(std::vector<int> triangles, int parent, int axis, int depth) {
    int node_index = numberOfNodes++;
    int next_axis = (axis + 1) % 3;
    Node temp(parent, node_index, nullptr, get_bounding_box(triangles), scene);
    nodes[node_index] = temp;
    Node &cur = nodes[node_index];
    std::vector<int> left, right;
    if (triangles.size() < 10 || depth > 15 || !split(triangles, left, right, axis)) {
        cur.num_of_triangles = triangles.size();
        cur.triangles = new int[triangles.size()];
        for (int i = 0; i < triangles.size(); ++i) {
            cur.triangles[i] = triangles[i];
        }
        return node_index;
    }
    if (left.size() != 0) {
        cur.left = build_tree(left, node_index, next_axis, depth + 1);
    }
    if (right.size() != 0) {
        cur.right = build_tree(right, node_index, next_axis, depth + 1);
    }
    return node_index;
}

Color KdTree::trace(Vector vector, int depth) {
    Vector *vectors = new Vector[depth + 1];
    int *triangles = new int[depth + 1];
    vector.normalize();
    vectors[0] = vector;
    triangles[0] = -1; // there is no triangle for primary vector
    int num = 1;
    for (; num <= depth; ++num) {
        vector = vectors[num - 1];
        vector.translateStartedPoint(FLT_EPSILON * 3);
        int triangle_index = get_triangle(vector);
        if (triangle_index == -1 || depth == num) {
            break;
        }
        vectors[num] = scene->getTriangles()[triangle_index].getReflectedVector(vectors[num - 1]);
        vectors[num].normalize();
        triangles[num] = triangle_index;
    }
    Color res(0, 0, 0);
    for (int i = num - 1; i >= 1; i--) {
        Point reflection_point = vectors[i].startPoint;
        Vector normal = scene->getTriangles()[triangles[i]].getNormal();
        normal.normalize();
        if (normal.isObtuse(vectors[i])) {
            normal = normal.mul(-1);
        }
        Vector to_viewer = vectors[i - 1].mul(-1);
        to_viewer.normalize();
        Material material = scene->getMaterial((scene->getTriangles()[triangles[i]]).materialCode);
        Color triangle_ilumination = Ia * material.ambient;
        for (int light = 0; light < numberOfLights; ++light) {
            Vector to_light = Vector(reflection_point, lights[light].point);
            to_light.normalize();
            // check if light is block out
            if (normal.isObtuse(to_light)) {
                continue;
            }
            Vector temp = to_light;
            temp.translateStartedPoint(FLT_EPSILON * 10);
            int index = get_triangle(temp);
            if (index == triangles[i]) {
                std::cout << "zle\n" << std::endl;
            }

            if (index != -1 && (scene->getTriangles()[index].getDist(to_light) <
                                lights[light].point.getDist(reflection_point))) { // fix this
                continue;
            }

            Vector from_light(lights[light].point, reflection_point);
            Vector from_light_reflected = scene->getTriangles()[triangles[i]].getReflectedVector(
                    from_light);
            from_light_reflected.normalize();
            triangle_ilumination +=
                    lights[light].Id * std::max(0.f, normal.dot(to_light)) * material.diffuse;
            triangle_ilumination +=
                    lights[light].Is * powf(std::max(0.f, to_viewer.dot(from_light_reflected)),
                                            material.specularExponent) * material.specular;
        }

        if (i < num - 1) {
            triangle_ilumination +=
                    res * powf(std::max(0.f, to_viewer.dot(normal)), material.specularExponent) *
                    material.
                            specular;
        }

        res = triangle_ilumination;
    }
    delete[] vectors;
    delete[] triangles;
    return res;
}

Box KdTree::get_bounding_box(std::vector<int> &triangles_) {
    Point min_point(FLT_MAX, FLT_MAX, FLT_MAX);
    float x = 0, y = 0, z = 0;
    for (auto &index : triangles_) {
        Triangle &triangle = scene->getTriangles()[index];
        min_point.x = std::min(min_point.x, triangle.getMinX());
        min_point.y = std::min(min_point.y, triangle.getMinY());
        min_point.z = std::min(min_point.z, triangle.getMinZ());
    }

    Point max_point(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (auto &index : triangles_) {
        Triangle &triangle = scene->getTriangles()[index];
        max_point.x = std::max(max_point.x, triangle.getMaxX());
        max_point.y = std::max(max_point.y, triangle.getMaxY());
        max_point.z = std::max(max_point.z, triangle.getMaxZ());
    }
    x = max_point.x - min_point.x;
    y = max_point.y - min_point.y;
    z = max_point.z - min_point.z;
    return Box(min_point, x, y, z);
}

bool KdTree::split(std::vector<int> &triangles, std::vector<int> &left, std::vector<int> &right,
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
    int mid_triangle = triangles[triangles.size() / 2];
    for (auto &triangle : triangles) {
        if (com(triangle, mid_triangle)) {
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

