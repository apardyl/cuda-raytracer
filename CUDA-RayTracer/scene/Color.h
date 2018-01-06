#ifndef RAY_TRACER_COLOR_H
#define RAY_TRACER_COLOR_H

struct Color {
    float red, green, blue;

    Color() = default;
    Color(float red, float green, float blue);
};

#endif // RAY_TRACER_COLOR_H
