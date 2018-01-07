#include "Color.h"

Color::Color() : red(0), green(0), blue(0) {
}

Color::Color(float red, float green, float blue)
    : red(red), green(green), blue(blue) {
}

Color& Color::operator+=(const Color &color) {
    this->red += color.red;
    this->green += color.green;
    this->blue += color.blue;
    return *this;
}

Color& Color::operator/=(const float &div) {
    this->red /= div;
    this->green /= div;
    this->blue /= div;
    return *this;
}

Color Color::operator*(const float &mul) const {
    Color res = Color(red * mul, green * mul, blue * mul);
    return res;
}

Color Color::operator/(const float &div) const {
    Color res = Color(red / div, green / div, blue / div);
    return res;
}
