#ifndef RAY_TRACER_COLOR_H
#define RAY_TRACER_COLOR_H

struct Color {
	float r, g, b;
	Color();
	Color(float r, float g, float  b);

	Color& operator+=(const Color& color);

	Color& operator/=(const float& div);

	Color operator*(const float& mul);

	Color operator/(const float& div);
};

#endif // RAY_TRACER_COLOR_H
