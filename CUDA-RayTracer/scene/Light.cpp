#include "Light.h"

Light::Light() = default;

Light::Light(Point point, Color Is, Color Id) {
	this->point = point;
	this->specular = Is;
	this->diffuse = Id;
}