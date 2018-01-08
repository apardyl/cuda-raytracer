#ifndef RAY_TRACER_COLOR_H
#define RAY_TRACER_COLOR_H

struct Color {
    float red, green, blue;

    Color();
    Color(float red, float green, float blue);

    Color& operator+=(const Color &color);

    Color& operator/=(const float &div);

    Color operator*(const float &mul) const;

    Color operator/(const float &div) const;
};

#endif // RAY_TRACER_COLOR_H
