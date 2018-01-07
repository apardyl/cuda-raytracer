#include "Light.h"


Light::Light() {}

Light::Light(Point point, Color Is, Color Id) {
	this->point = point;
	this->Is = Is;
	this->Id = Id;
}
