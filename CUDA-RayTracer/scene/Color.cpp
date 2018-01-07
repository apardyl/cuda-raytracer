#include "Color.h"

Color::Color() {}

Color::Color(float r, float g, float  b) {
	this->r = r;
	this->g = g;
	this->b = b;
}

Color& Color::operator+=(const Color& color) {
	this->r += color.r;
	this->g += color.g;
	this->b += color.b;
	return *this;
}

Color& Color::operator/=(const float& div) {
	this->r /= div;
	this->g /= div;
	this->b /= div;
	return *this;
}

Color Color::operator*(const float& mul) {
	Color res = Color(r*mul, g*mul, b*mul);
	return res;
}

Color Color::operator/(const float& div) {
	Color res = Color(r / div, g / div, b / div);
	return res;
}

