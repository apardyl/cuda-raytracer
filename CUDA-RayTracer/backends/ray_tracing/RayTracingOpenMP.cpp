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


const int N = (int)1e6 + 5;
const float inf = 1e9;
const float eps = 1e-6;
const int BYTES_PER_PIXEL = 3;

int num_of_triangles = 0;
int num_of_nodes = 0;
int num_of_lights = 0;

Material THE_MATERIAL(Color(0.2, 0, 0), Color(0.5, 0, 0), Color(0.6, 0, 0), 0.2, 1);

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
	float width, height;
	Resolution resolution;
	Point focus_point = Point(0, 0, -1);
	int num_of_samples = 1024;
	Color active_pixel_sensor[3003][3003];
	Random random;

	Camera(float width, float height, Resolution resolution, int num_of_samples) {
		this->width = width;
		this->height = height;
		this->resolution = resolution;
		this->num_of_samples = num_of_samples;
		for (int i = 0; i < resolution.height; ++i) {
			for (int j = 0; j < resolution.width; ++j) {
				active_pixel_sensor[i][j] = Color(0, 0, 0);
			}
		}
	}

	Vector get_random_vector(int row, int column) { // not implemented
		return Vector(Point(0, 0, 0), 0, 0, 0);
	}

	Vector get_primary_vector(int row, int column) { // row from [0,..., resolution.height-1]  column from [0, resolution.width-1]
		float pixel_width = width / resolution.width;
		float pixel_height = height / resolution.height;
		float x_cord = (-width / 2) + pixel_width * (0.5 + column);
		float y_cord = (-height / 2) + pixel_height * (0.5 + row);
		return Vector(focus_point, Point(x_cord, y_cord, 0));
	}

	void update(int row, int column, Color color) {
		active_pixel_sensor[row][column] += color;
	}

	Color get_pixel_color(int i, int j) {
		return active_pixel_sensor[i][j] / (float)num_of_samples;
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
		if (normal.getAngle(vectors[i]) > PI) 
		{
			normal = normal.mul(-1);
		}
		Vector to_viewer = vectors[i - 1].mul(-1);
		to_viewer.normalize();
        Material material = THE_MATERIAL;
		Color triangle_ilumination = Ia * material.ambient.red;
		for (int light = 0; light < num_of_lights; ++light)
		{
			Vector to_light = Vector(reflection_point, lights[light].point);
			to_light.normalize();
			// check if light is block out
			//to_light.startPoint = to_light.startPoint.translate(vectors[i].mul((FLT_EPSILON ) / vectors[i].len()));
			//if (get_triangle(to_light) != -1) // fix this
			//	continue;
			//
			Vector from_light(lights[light].point, reflection_point);
			Vector from_light_reflected = global_triangles[triangles[i]].getReflectedVector(from_light);
			from_light_reflected.normalize();
			triangle_ilumination += lights[light].Id*std::max(0.f, (normal.dot(to_light)))*material.diffuse.red;
			triangle_ilumination += lights[light].Is*powf(std::max(0.f, to_viewer.dot(from_light_reflected)), material.specularExponent)*material.specular.red;
		}

		if (i < num - 1)
		{
			triangle_ilumination += res * powf(std::max(0.f, to_viewer.dot(normal)), material.specularExponent)*material.specular.red;
		}

		res = triangle_ilumination;
	}
	delete[] vectors;
	delete[] triangles;
	return res;
}

Image RayTracingOpenMP::render() {
	nodes = new Node[N];
	lights = new Light[10];
	Light light(Point(0, 0, 0.5), Color(100, 100, 100), Color(100, 100, 100));
	lights[0] = light;
	num_of_lights++;
	
	global_triangles = new Triangle[N];
	global_triangles[0] = Triangle(Point(-1.25, -0.81, 0), Point(0.79, -0.81, 0), Point(0, 0, 1.5));
	global_triangles[1] = Triangle(Point(-0.376859, 0.353287, -0.324435)
		, Point(1.623141, 0.353287, -0.324435), Point(-0.376859, 0.353287, 1.675565));
	global_triangles[2] = Triangle(Point(1.22, 1.15, 0)
		, Point(0, 0, 1), Point(-1.35, 1.28, 0));
		
	//global_triangles[0] = Triangle(Point(4.64, -2.6, 2), Point(7.92, 1.67, 2), Point(6.21, -5.27, 2), A);
	//global_triangles[1] = Triangle(Point(-7.08, 7.7, 2)
	//	, Point(1, 4, 2), Point(0, 0, 2), A);
	//global_triangles[2] = Triangle(Point(1.22, 1.15, 2)
	//	, Point(0, 0, 2), Point(-1.35, 1.28, 2), B);
	num_of_triangles = 3;
	std::vector<int> triangles = {0, 1, 2};
	build_tree(triangles, -1, 0, 10);
	Resolution resolution = Resolution(width, height);
	Camera camera(2, 2, resolution, 1);
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			Vector vector = camera.get_primary_vector(i, j);
			Color color = trace(vector, 20);
			camera.update(i, j, color);
		}
	}
	delete[] nodes;
	delete[] global_triangles;
	delete[] lights;
	// return Image
	data = new byte[width * height * BYTES_PER_PIXEL];
	for (int i = 0; i < resolution.height; ++i)
	{
		for (int j = 0; j < resolution.width; ++j)
		{
			Color color = camera.get_pixel_color(resolution.height - i - 1, j);
			int y = height - 1 - i;
			int x = j;
			data[(width * y + x) * BYTES_PER_PIXEL] = color.red;
			data[(width * y + x) * BYTES_PER_PIXEL + 1] = color.green;
			data[(width * y + x) * BYTES_PER_PIXEL + 2] = color.blue;
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
