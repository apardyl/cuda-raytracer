#include "RayTracingOpenMP.h"
#include "../../scene/Color.h"
#include "../../scene/Vector.h"
#include "../../scene/Point.h"
#include "../../scene/Triangle.h"
#include "../../scene/Material.h"
#include <utility>
#include <vector>
#include <algorithm>
#include <limits>
#include <iostream>
#include <cmath>
#define PI 3.14159265
#include <cfloat>

const int N = (int)1e6 + 5;
const float inf = 1e9;
const float eps = 1e-6;
const int BYTES_PER_PIXEL = 3;

int num_of_triangles = 0;
int num_of_nodes = 0;
int num_of_lights = 0;

Material * materials;

struct Triangle;
struct Node;
struct Light;

Triangle * global_triangles = NULL;
Node * nodes = NULL;
Light * lights = NULL;

Color Ia(0.2, 0.2, 0.2); // ambient intensities 

struct Light {
	Point point;
	Color Is, Id;

	Light() {}

	Light(Point point, Color Is, Color Id) {
		this->point = point;
		this->Is = Is;
		this->Id = Id;
	}
};

struct Box {
	Point point;
	float x, y, z;

	Box() {}

	Box(Point point, float x, float y, float z) {
		this->point = point;
		this->x = x;
		this->y = y;
		this->z = z;
	}

	float get_dist(Vector vector) // not implemented
	{
		return 0;
	}

	Point getMin() const {
		return point;
	}

	Point getMax() const {
		Point p;
		p.x = point.x + x;
		p.y = point.y + y;
		p.z = point.z + z;
		return p;
	}

	bool is_intersecting(Vector vector) {
		Point pMin = getMin();
		Point pMax = getMax();

		float txMin = (pMin.x - vector.startPoint.x) / vector.x;
		float txMax = (pMax.x - vector.startPoint.x) / vector.x;

		if (txMin > txMax) {
			std::swap(txMin, txMax);
		}

		float tyMin = (pMin.y - vector.startPoint.y) / vector.y;
		float tyMax = (pMax.y - vector.startPoint.y) / vector.y;

		if (tyMin > tyMax) {
			std::swap(tyMin, tyMax);
		}

		if ((txMin > tyMax) || (tyMin > txMax)) {
			return false;
		}

		if (tyMin > txMin) {
			txMin = tyMin;
		}

		if (tyMax < txMax) {
			txMax = tyMax;
		}

		float tzMin = (pMin.z - vector.startPoint.z) / vector.z;
		float tzMax = (pMax.z - vector.startPoint.z) / vector.z;

		if (tzMin > tzMax) {
			std::swap(tzMin, tzMax);
		}

		if ((txMin > tzMax) || (tzMin > txMax)) {
			return false;
		}

		if (tzMin > txMin) {
			txMin = tzMin;
		}

		if (tzMax < txMax) {
			txMax = tzMax;
		}

		return true;
	}

	Box& operator=(const Box &other) {
		if (this != &other)
		{
			this->x = other.x;
			this->y = other.y;
			this->z = other.z;
			this->point = other.point;
		}
		return *this;
	}
};

struct Node {
	int parent, left, right, my_index;
	int * triangles = NULL;
	int num_of_triangles = 0;
	Box bounding_box;

	Node() {}

	Node(int parent, int my_index, int * triangles, Box bounding_box) {
		this->parent = parent;
		this->left = -1;
		this->right = -1;
		this->my_index = my_index;
		this->triangles = triangles;
		this->bounding_box = bounding_box;
	}

	int get_minimal_triangle(Vector &vector) // if there is no such triangle return -1
	{
		int best_index = -1;
		float best = inf;
		if (left == -1 && right == -1)
		{
			for (int i = 0; i < num_of_triangles; ++i)
			{
				float dist = global_triangles[this->triangles[i]].getDist(vector);
				if (dist != -1 && dist < best)
				{
					best_index = triangles[i];
					best = dist;
				}
			}
		}
		return best_index;
	}

	bool is_leaf() {
		return (left == right);
	}

	~Node() {
		if (triangles != NULL)
		{
			delete[] triangles;
		}
	}
};

struct Stack {
	int tab[60];
	int size = 0;

	Stack() {}

	void add_element(int x) {
		tab[size++] = x;
	}

	int top() {
		return tab[size - 1];
	}

	void pop() {
		size--;
	}
};

/// Comparators

bool com_by_x(const int & a, const int & b) {
	return global_triangles[a].getMidpoint().x < global_triangles[b].getMidpoint().x;
}

bool com_by_y(const int & a, const int & b) {
	return global_triangles[a].getMidpoint().y < global_triangles[b].getMidpoint().y;
}

bool com_by_z(const int & a, const int & b) {
	return global_triangles[a].getMidpoint().z < global_triangles[b].getMidpoint().z;
}

///

bool split(std::vector<int> &triangles, std::vector<int> &left, std::vector<int> &right, int axis) {
	if (triangles.size() <= 1) return false;
	bool(*com) (const int & a, const int & b) = NULL;
	switch (axis)
	{
	case 0:
		com = &com_by_x;
		break;
	case 1:
		com = &com_by_y;
		break;
	case 2:
		com = &com_by_z;
		break;
	}

	std::sort(triangles.begin(), triangles.end(), com);
	int mid_triangle = triangles[triangles.size() / 2];
	for (auto & triangle : triangles)
	{
		if (com(triangle, mid_triangle))
		{
			left.push_back(triangle);
		} else
		{
			right.push_back(triangle);
		}
	}

	return true;
}

Box get_bounding_box(std::vector<int> &triangles_) {
	Point min_point(inf, inf, inf);
	float x = 0, y = 0, z = 0;
	for (auto &index : triangles_)
	{
		Triangle &triangle = global_triangles[index];
		min_point.x = std::min(min_point.x, triangle.getMinX());
		min_point.y = std::min(min_point.y, triangle.getMinY());
		min_point.z = std::min(min_point.z, triangle.getMinZ());
	}

	Point max_point(-inf, -inf, -inf);
	for (auto &index : triangles_)
	{
		Triangle &triangle = global_triangles[index];
		max_point.x = std::max(max_point.x, triangle.getMaxX());
		max_point.y = std::max(max_point.y, triangle.getMaxY());
		max_point.z = std::max(max_point.z, triangle.getMaxZ());
	}
	x = max_point.x - min_point.x;
	y = max_point.y - min_point.y;
	z = max_point.z - min_point.z;
	return Box(min_point, x, y, z);
}

struct Resolution {
	int width, height;

	Resolution() {}

	Resolution(int width, int height) {
		this->width = width;
		this->height = height;
	}
};

struct Random {

};

struct Camera {
    Point location;
    Point rotation;

	float width, height;
	Resolution resolution;
	int num_of_samples = 1024;
	std::unique_ptr<Color[]> active_pixel_sensor;
	Random random;

	Camera(Point location, Point rotation, float width, float height, Resolution resolution,
		   int num_of_samples) :
			location(location),
			rotation(rotation),
			width(width),
			height(height),
			resolution(resolution),
			num_of_samples(num_of_samples),
			active_pixel_sensor(std::make_unique<Color[]>(resolution.width * resolution.height)) {
		for (int y = 0; y < resolution.height; ++y) {
			for (int x = 0; x < resolution.width; ++x) {
				getActivePixelSensor(x, y) = Color(0, 0, 0);
			}
		}
	}

	Color& getActivePixelSensor(int x, int y) const {
		return active_pixel_sensor[resolution.width * y + x];
	}

	Vector get_random_vector(int x, int y) const { // not implemented
		return Vector(Point(0, 0, 0), 0, 0, 0);
	}

	Vector get_primary_vector(int x, int y) const {
		float pixel_width = width / resolution.width;
		float pixel_height = height / resolution.height;
		float x_cord = (-width / 2) + pixel_width * (0.5 + x);
		float y_cord = (-height / 2) + pixel_height * (0.5 + y);
        Vector vector = Vector(Point(0,0,0), Point(y_cord, x_cord, -1))
                .rotateX(rotation.x)
                .rotateY(rotation.y)
                .rotateZ(rotation.z);
        vector.startPoint = location;
        return vector;
	}

	void update(int x, int y, Color color) {
		getActivePixelSensor(x, y) += color;
	}

	Color get_pixel_color(int x, int y) const {
        return getActivePixelSensor(x, y) / (float) num_of_samples;
    }
};

int build_tree(std::vector<int> triangles, int parent, int axis, int depth) {
	int node_index = num_of_nodes++;
	int next_axis = (axis + 1) % 3;
	Node temp(parent, node_index, NULL, get_bounding_box(triangles));
	nodes[node_index] = temp;
	Node &cur = nodes[node_index];
	std::vector<int> left, right;
	if (depth > 30 || !split(triangles, left, right, axis))
	{
		cur.num_of_triangles = triangles.size();
		cur.triangles = new int[triangles.size()];
		for (int i = 0; i < triangles.size(); ++i)
		{
			cur.triangles[i] = triangles[i];
		}
		return node_index;
	} 
	else
	{
		if (left.size() != 0)
		{
			cur.left = build_tree(left, node_index, next_axis, depth + 1);
		}
		if (right.size() != 0)
		{
			cur.right = build_tree(right, node_index, next_axis, depth + 1);
		}
	}
	return node_index;
}

int get_triangle(Vector &vector) // get triangle which have collison with vector // if there isn't any triangle return -1
{
	int ans = -1;
	float best_distance = inf;
	Stack stack;
	stack.add_element(0);
	while (stack.size > 0)
	{
		int cur = stack.top();
		stack.pop();
		if (nodes[cur].bounding_box.is_intersecting(vector))
		{
			if (nodes[cur].is_leaf())
			{
				int index_of_best_triangle = nodes[cur].get_minimal_triangle(vector);
				if (index_of_best_triangle != -1)
				{
					float distance = global_triangles[index_of_best_triangle].getDist(vector);
					if (best_distance > distance)
					{
						best_distance = distance;
						ans = index_of_best_triangle;
					}
				}
			} else
			{
				if (nodes[cur].left != -1)
					stack.add_element(nodes[cur].left);
				if (nodes[cur].right != -1)
					stack.add_element(nodes[cur].right);
			}
		}
	}
	return ans;
}

Color trace(Vector vector, int depth) {
	Vector * vectors = new Vector[depth];
	int * triangles = new int[depth];
	vector.normalize();
	vectors[0] = vector;
	triangles[0] = -1; // there is no triangle for primary vector
	int num = 1;
	for (; num <= depth; ++num)
	{
		vector = vectors[num - 1];
		vector.translateStartedPoint(FLT_EPSILON * 3);
		int triangle_index = get_triangle(vector);
		if (triangle_index == -1 || depth == num)
		{
			break;
		}
		vectors[num] = global_triangles[triangle_index].getReflectedVector(vectors[num - 1]);
		vectors[num].normalize();
		triangles[num] = triangle_index;
	}
	Color res(0, 0, 0);
	for (int i = num-1; i >= 1; i--)
	{
		Point reflection_point = vectors[i].startPoint;
		Vector normal = global_triangles[triangles[i]].getNormal(); 
		normal.normalize();
		if (normal.isObtuse(vectors[i]))
		{
			normal = normal.mul(-1);
		}
		Vector to_viewer = vectors[i - 1].mul(-1);
		to_viewer.normalize();
        Material material = materials[global_triangles[triangles[i]].materialCode];
		Color triangle_ilumination = Ia * material.ambient;
		for (int light = 0; light < num_of_lights; ++light)
		{
			Vector to_light = Vector(reflection_point, lights[light].point);
			to_light.normalize();
			// check if light is block out
			if (normal.isObtuse(to_light))
				continue;
			Vector temp = to_light;
			temp.translateStartedPoint(FLT_EPSILON*10);
			int index = get_triangle(temp);
			if (index == triangles[i])
			{
				std::cout << "zle\n" << std::endl;
			}
				
			if (index != -1 && (global_triangles[index].getDist(to_light) < lights[light].point.getDist(reflection_point))) // fix this
			{
				continue;
			}
			
			//
			Vector from_light(lights[light].point, reflection_point);
			Vector from_light_reflected = global_triangles[triangles[i]].getReflectedVector(from_light);
			from_light_reflected.normalize();
			triangle_ilumination += lights[light].Id*std::max(0.f, normal.dot(to_light))*material.diffuse;
			triangle_ilumination += lights[light].Is*powf(std::max(0.f, to_viewer.dot(from_light_reflected)), material.specularExponent)*material.specular;
		}

		if (i < num - 1)
		{
			triangle_ilumination += res * powf(std::max(0.f, to_viewer.dot(normal)), material.specularExponent)*material.specular;
		}

		res = triangle_ilumination;
	}
	delete[] vectors;
	delete[] triangles;
	return res;
}

Image RayTracingOpenMP::render() {
    materials = scene.get()->getMaterials();
	nodes = new Node[N];
	lights = new Light[10];
	Light light(Point(0, 0, -1), Color(255, 255, 255), Color(255, 255, 255));
	lights[0] = light;
	num_of_lights++;

	global_triangles = scene->getTriangles();
	num_of_triangles = scene->trianglesCount;
	std::vector<int> triangles(num_of_triangles);
	for (int i = 0; i < num_of_triangles; ++i) {
		triangles[i] = i;
	}
	build_tree(triangles, -1, 0, 10);
	Resolution resolution = Resolution(width, height);
	Camera camera(Point(0, 0, -1), Point(0, static_cast<float>(M_PI), 0), 2, 2, resolution, 1);
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			Vector vector = camera.get_primary_vector(x, y);
			Color color = trace(vector, 20);
			camera.update(x, y, color);
		}
	}
	delete[] nodes;
	delete[] lights;
	// return Image
	data = new byte[width * height * BYTES_PER_PIXEL];
	for (int y = 0; y < resolution.height; ++y)
	{
		for (int x = 0; x < resolution.width; ++x)
		{
			Color color = camera.get_pixel_color(x, y);
			data[(width * y + x) * BYTES_PER_PIXEL] = std::min(color.red, (float)255);
			data[(width * y + x) * BYTES_PER_PIXEL + 1] = std::min(color.green, (float) 255);
			data[(width * y + x) * BYTES_PER_PIXEL + 2] = std::min(color.blue, (float) 255);
		}
	}
	return Image(resolution.width, resolution.height, data);
}

RayTracingOpenMP::~RayTracingOpenMP() {
	delete[] data;
}

RayTracingOpenMP::RayTracingOpenMP() { }

void RayTracingOpenMP::setResolution(unsigned width, unsigned height) {
	this->width = width;
	this->height = height;
}

void RayTracingOpenMP::setSoftShadows(bool var) { // not implemented

}
