#ifndef RAY_TRACER_IMAGE_H
#define RAY_TRACER_IMAGE_H

typedef unsigned char byte;

struct Image {
    unsigned width;
    unsigned height;

    byte *pixelData;

    Image(unsigned int width, unsigned int height, byte *pixelData) :
            width(width), height(height), pixelData(pixelData) {}
};

#endif //RAY_TRACER_IMAGE_H
