#include "RayTracingOpenMP.h"
#include <utility>
#include <vector>
#include <algorithm>
#include <limits>
#include <iostream>
#include <cmath>

const int N = (int)1e6 + 5;
const float inf = 1e9;
const float eps = 1e-6;

int num_of_triangles = 0;
int num_of_nodes = 0;
int num_of_lights = 0;

struct Triangle;
struct Node;
struct Vector;
struct Light;
struct Color;

Triangle * global_triangles = NULL;
Node * nodes = NULL;
Light * lights = NULL;

struct Point {
	float x, y, z;

	Point() {}

	Point(float x, float y, float z) {
		this->x = x;
		this->y = y;
		this->z = z;
	}

	float get_dist(float x, float y, float z) {
		float u = this->x - x;
		float v = this->y - y;
		float m = this->z - z;
		return sqrt((u*u) + v * v + m * m);
	}

	Point& operator=(const Point &other) {
		if (this != &other)
		{
			this->x = other.x;
			this->y = other.y;
			this->z = other.z;
		}
		return *this;
	}

	Point translate(Vector vector);
};

struct Vector {
	Point point;
	float x, y, z;

	Vector() {}

	Vector(Point point, float x, float y, float z) {
		this->point = point;
		this->x = x;
		this->y = y;
		this->z = z;
	}

	Vector(Point a, Point b) {
		this->point = a;
		this->x = b.x - a.x;
		this->y = b.y - a.y;
		this->z = b.z - a.z;
	}

	Vector mul(float scl) {
		return Vector(point, this->x*scl, this->y*scl, this->z*scl);
	}

	Vector add(Vector other) {
		return Vector(point, x + other.x, y + other.y, z + other.z);
	}

	Vector cross_product(Vector vec) {
		int vec_x = y * vec.z - vec.y * z;
		int vec_y = z * vec.x - vec.z * x;
		int vec_z = x * vec.y - vec.x *	y;
		Vector res = Vector(point, vec_x, vec_y, vec_z);
		return res;
	}

	float dot(Vector vector) {
		return x * vector.x + y * vector.y + z * vector.z;
	}

	float len() {
		return sqrt(x * x + y * y + z * z);
	}

	void normalize() {
		float div = len();
		if (div != 0)
		{
			x /= div;
			y /= div;
			z /= div;
		}
	}
};


struct Color {
	float r, g, b;
	Color() {}
	Color(float r, float g, float  b) {
		this->r = r;
		this->g = g;
		this->b = b;
	}

	Color& operator+=(const Color& color) {
		this->r += color.r;
		this->g += color.g;
		this->b += color.b;
		return *this;
	}

	Color& operator/=(const float& div) {
		this->r /= div;
		this->g /= div;
		this->b /= div;
		return *this;
	}

	Color operator*(const float& mul) {
		Color res = Color(r*mul, g*mul, b*mul);
		return res;
	}

	Color operator/(const float& div) {
		Color res = Color(r / div, g / div, b / div);
		return res;
	}
};

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

struct Material {
	float Ks, Kd, Ka, alfa;

	Material() {}

	Material(float Ks, float Kd, float Ka, float alfa) {
		this->Ks = Ks;
		this->Kd = Kd;
		this->Ka = Ka;
		this->alfa = alfa;
	}
};

struct Triangle {
	Point x, y, z;
	Material material;

	Triangle() {}

	Triangle(Point a, Point b, Point c, Material material) {
		x = a;
		y = b;
		z = c;
		this->material = material;
	}

	Point get_midpoint() const {
		float mid_x = (x.x + y.x + z.x) / 3;
		float mid_y = (x.y + y.y + z.y) / 3;
		float mid_z = (x.z + y.z + z.z) / 3;
		Point res(mid_x, mid_y, mid_z);
		return  res;
	}

	float get_dist(Vector vector) // if the is no intersection return -1
	{
		vector.normalize();
		Vector a_b(x, y);
		Vector a_c(x, z);
		Vector normal = a_b.cross_product(a_c);
		Vector origin(Point(0, 0, 0), vector.point.x, vector.point.y, vector.point.z);
		Vector A(Point(0, 0, 0), x.x, x.y, x.z);
		if (fabs(normal.dot(vector)) < eps)  // if triangle is parallel to vector return -1
			return -1;
		float D = -normal.dot(A);
		float dist_to_plane = -(normal.dot(origin) + D) / (normal.dot(vector));
		if (dist_to_plane < 0) // vector is directed in opposite direction
			return -1;
		// check if intersection point is inside the triangle
		Point p(vector.point.translate(vector.mul(dist_to_plane)));
		Vector edge_a_b(x, y);
		Vector edge_b_c(y, z);
		Vector edge_c_a(z, x);
		Vector a_p(x, p);
		Vector b_p(y, p);
		Vector c_p(z, p);
		bool on_left_a_b = normal.dot(edge_a_b.cross_product(a_p)) > 0;
		bool on_left_b_c = normal.dot(edge_b_c.cross_product(b_p)) > 0;
		bool on_left_c_a = normal.dot(edge_c_a.cross_product(c_p)) > 0;
		if (on_left_a_b && on_left_b_c && on_left_c_a)
		{
			return dist_to_plane;
		}
		return -1;
	}

	Vector get_reflected_vector(Vector vector) {
		vector.normalize();
		Vector a_b(x, y);
		Vector a_c(x, z);
		Vector normal = a_b.cross_product(a_c);
		normal.normalize();
		Vector res = vector.add(normal.mul((-2)*vector.dot(normal)));
		res.normalize();
		return res;
	}

	Vector get_normal() {
		Vector a_b(x, y);
		Vector a_c(x, z);
		Vector normal = a_b.cross_product(a_c);
		return normal;
	}

	Point get_intersection_point(Vector vector) // not implemented
	{
		return x;
	}

	float get_min_x() {
		return std::min(x.x, std::min(y.x, z.x));
	}

	float get_min_y() {
		return std::min(x.y, std::min(y.y, z.y));
	}

	float get_min_z() {
		return std::min(x.z, std::min(y.z, z.z));
	}

	float get_max_x() {
		return std::max(x.x, std::max(y.x, z.x));
	}

	float get_max_y() {
		return std::max(x.y, std::max(y.y, z.y));
	}

	float get_max_z() {
		return std::max(x.z, std::max(y.z, z.z));
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

		float txMin = (pMin.x - vector.point.x) / vector.x;
		float txMax = (pMax.x - vector.point.x) / vector.x;

		if (txMin > txMax) {
			std::swap(txMin, txMax);
		}

		float tyMin = (pMin.y - vector.point.y) / vector.y;
		float tyMax = (pMax.y - vector.point.y) / vector.y;

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

		float tzMin = (pMin.z - vector.point.z) / vector.z;
		float tzMax = (pMax.z - vector.point.z) / vector.z;

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
				float dist = global_triangles[this->triangles[i]].get_dist(vector);
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
	return global_triangles[a].get_midpoint().x < global_triangles[b].get_midpoint().x;
}

bool com_by_y(const int & a, const int & b) {
	return global_triangles[a].get_midpoint().y < global_triangles[b].get_midpoint().y;
}

bool com_by_z(const int & a, const int & b) {
	return global_triangles[a].get_midpoint().z < global_triangles[b].get_midpoint().z;
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
		min_point.x = std::min(min_point.x, triangle.get_min_x());
		min_point.y = std::min(min_point.y, triangle.get_min_y());
		min_point.z = std::min(min_point.z, triangle.get_min_z());
	}

	Point max_point(-inf, -inf, -inf);
	for (auto &index : triangles_)
	{
		Triangle &triangle = global_triangles[index];
		max_point.x = std::max(max_point.x, triangle.get_max_x());
		max_point.y = std::max(max_point.y, triangle.get_max_y());
		max_point.z = std::max(max_point.z, triangle.get_max_z());
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
	Point focus_point = Point(0, 0, 1);
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
		float x_cord = -width / 2 + pixel_width * (0.5 + column);
		float y_cord = width / 2 - pixel_width * (0.5 + row);
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
					float distance = global_triangles[index_of_best_triangle].get_dist(vector);
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
	vectors[0] = vector;
	triangles[0] = -1; // there is no triangle for primary vector
	int num = 1;
	for (; num < depth; ++num)
	{
		int triangle_index = get_triangle(vector);
		if (triangle_index == -1 || num == depth - 1)
		{
			num--;
			break;
		}
		vectors[num] = global_triangles[triangle_index].get_reflected_vector(vectors[num - 1]);
		triangles[num] = triangle_index;
	}
	Color res(0, 0, 0);
	for (int i = num - 1; i >= 1; i--)
	{
		Point reflection_point = vectors[i].point;
		Vector normal = global_triangles[triangles[i]].get_normal();
		normal.normalize();
		Vector to_viewer = vectors[i - 1].mul(-1);
		to_viewer.normalize();
		Material material = global_triangles[triangles[i]].material;
		Color triangle_ilumination = Ia * material.Ka;
		for (int light = 0; light < num_of_lights; ++light)
		{
			Vector to_light = Vector(reflection_point, lights[light].point);
			to_light.normalize();
			// check if light is block out
			if (get_triangle(to_light) != -1)
				continue;
			//
			Vector from_light(lights[light].point, reflection_point);
			Vector from_light_reflected = global_triangles[triangles[i]].get_reflected_vector(from_light);
			from_light_reflected.normalize();
			triangle_ilumination += lights[light].Id*(normal.dot(to_light))*material.Kd;
			triangle_ilumination += lights[light].Is*powf(to_viewer.dot(from_light_reflected), material.alfa)*material.Ks;
		}

		if (i < num - 1)
		{
			triangle_ilumination += res * powf(to_viewer.dot(to_viewer), material.alfa)*material.Ks;
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
	Light light(Point(7, -7, 7), Color(5, 5, 5), Color(8, 8, 8));
	lights[0] = light;
	num_of_lights++;
	Material A(0.2, 0.5, 0.6, 0.2);
	Material B(0.8, 0.3, 0.1, 0.5);
	global_triangles = new Triangle[N];
	global_triangles[0] = Triangle(Point(-1.25, -0.81, 0), Point(0.79, -0.81, 0), Point(0, 0, 1.5), A);
	global_triangles[1] = Triangle(Point(-0.376859, 0.353287, -0.324435)
		, Point(1.623141, 0.353287, -0.324435), Point(-0.376859, 0.353287, 1.675565), A);
	global_triangles[2] = Triangle(Point(1.22, 1.15, 0)
		, Point(0, 0, 1), Point(-1.35, 1.28, 0), B);
	num_of_triangles = 3;
	std::vector<int> triangles = { 0,1,2 };
	build_tree(triangles, -1, 0, 10);
	Camera camera(400, 400, Resolution(400, 400), 1);
	for (int i = 0; i < 400; ++i) {
		for (int j = 0; j < 400; ++j) {
			Vector vector = camera.get_primary_vector(i, j);
			Color color = trace(vector, 20);
			camera.update(i, j, color);
		}
	}
	delete[] nodes;
	delete[] global_triangles;
	delete[] lights;
	// return Image
	cudaMallocHost(&data, sizeof(byte) * width * height * BYTES_PER_PIXEL);
	for (int i = 0; i < resolution.height; ++i)
	{
		for (int j = 0; j < resolution.width; ++j)
		{
			Color color = camera.get_pixel_color(resolution.height - i - 1, j);
			data[(width * y + x) * BYTES_PER_PIXEL] = color.r;
			data[(width * y + x) * BYTES_PER_PIXEL + 1] = color.g;
			data[(width * y + x) * BYTES_PER_PIXEL + 2] = color.b;
		}
	}
	return Image(resolution.width, resolution.height, data);
}

Point Point::translate(Vector vector) {
	return Point(x + vector.x, y + vector.y, z + vector.z);
}

RayTracingOpenMP::~RayTracingOpenMP() {
	cudaFree(data);
}