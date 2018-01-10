#ifndef RAY_TRACER_IMAGE_H
#define RAY_TRACER_IMAGE_H
#include "scene/Color.h"

typedef unsigned char byte;

struct Image {
    unsigned width;
    unsigned height;

    Color *pixelData = nullptr;

    Image(unsigned int width, unsigned int height, Color *pixelData);

    Image(Image const &image);

    Image(Image &&image) noexcept;

    ~Image();
};

#endif //RAY_TRACER_IMAGE_H
