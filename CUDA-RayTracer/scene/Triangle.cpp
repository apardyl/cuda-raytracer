#include "Triangle.h"
#include "Material.h"
#include "Vector.h"
#include <algorithm>


const float inf = 1e9;
const float eps = 1e-6;

Triangle::Triangle() {}

Triangle::Triangle(Point a, Point b, Point c, Material material) {
	x = a;
	y = b;
	z = c;
	this->material = material;
}

Point Triangle::get_midpoint() const {
	float mid_x = (x.x + y.x + z.x) / 3;
	float mid_y = (x.y + y.y + z.y) / 3;
	float mid_z = (x.z + y.z + z.z) / 3;
	Point res(mid_x, mid_y, mid_z);
	return  res;
}

float Triangle::get_dist(Vector vector) // if the is no intersection return -1
{
	vector.normalize();
	Vector a_b(x, y);
	Vector a_c(x, z);
	Vector normal = a_b.cross_product(a_c);
	Vector origin(Point(0, 0, 0), vector.startPoint.x, vector.startPoint.y, vector.startPoint.z);
	Vector A(Point(0, 0, 0), x.x, x.y, x.z);
	if (fabs(normal.dot(vector)) < eps)  // if triangle is parallel to vector return -1
		return -1;
	float D = -normal.dot(A);
	float dist_to_plane = -(normal.dot(origin) + D) / (normal.dot(vector));
	if (dist_to_plane < 0) // vector is directed in opposite direction
		return -1;
	// check if intersection point is inside the triangle
	Point p(vector.startPoint.translate(vector.mul(dist_to_plane)));
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

Vector Triangle::get_reflected_vector(Vector vector) {
	vector.normalize();
	Vector a_b(x, y);
	Vector a_c(x, z);
	Vector normal = a_b.cross_product(a_c);
	normal.normalize();
	Vector res = vector.add(normal.mul((-2)*vector.dot(normal)));
	res.normalize();
	return res;
}

Vector Triangle::get_normal() {
	Vector a_b(x, y);
	Vector a_c(x, z);
	Vector normal = a_b.cross_product(a_c);
	return normal;
}

Point Triangle::get_intersection_point(Vector vector) // not implemented
{
	return x;
}

float Triangle::get_min_x() {
	return std::min(x.x, std::min(y.x, z.x));
}

float Triangle::get_min_y() {
	return std::min(x.y, std::min(y.y, z.y));
}

float Triangle::get_min_z() {
	return std::min(x.z, std::min(y.z, z.z));
}

float Triangle::get_max_x() {
	return std::max(x.x, std::max(y.x, z.x));
}

float Triangle::get_max_y() {
	return std::max(x.y, std::max(y.y, z.y));
}

float Triangle::get_max_z() {
	return std::max(x.z, std::max(y.z, z.z));
}

Triangle& Triangle::operator=(const Triangle &other) {
	if (this != &other)
	{
		this->x = other.x;
		this->y = other.y;
		this->z = other.z;
		this->material = other.material;
	}
	return *this;
}