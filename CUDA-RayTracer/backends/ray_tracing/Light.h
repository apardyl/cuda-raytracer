#pragma once
#include "scene/Color.h"
#include "scene/Point.h"

struct Light {
	Point point;
	Color Is, Id;

	Light();

	Light(Point point, Color Is, Color Id);
};
