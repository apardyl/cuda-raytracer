#ifndef RAY_TRACER_TRIANGLE_H
#define RAY_TRACER_TRIANGLE_H

#include "Point.h"
#include "Material.h"

struct Triangle {
	Point x, y, z;
	Material material;

	Triangle();

	Triangle(Point a, Point b, Point c, Material material);

	Point get_midpoint() const;

	float get_dist(Vector vector);

	Vector get_reflected_vector(Vector vector);

	Vector get_normal();

	Point get_intersection_point(Vector vector);

	float get_min_x();

	float get_min_y();

	float get_min_z();

	float get_max_x();

	float get_max_y();

	float get_max_z();

	Triangle& operator=(const Triangle &other);
};

#endif // RAY_TRACER_TRIANGLE_H
