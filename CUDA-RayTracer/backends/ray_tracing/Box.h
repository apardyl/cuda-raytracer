#pragma once
#include <algorithm>
#include "../../scene/Point.h"
#include "../../scene/Vector.h"


struct Box {
	Point point;
	float x, y, z;

	Box();

	Box(Point point, float x, float y, float z);

	float get_dist(Vector vector); // not implemented

	Point getMin() const;

	Point getMax() const;

	bool is_intersecting(Vector vector);

	Box& operator=(const Box& other);
};
