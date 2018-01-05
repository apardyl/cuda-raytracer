#ifndef RAY_TRACER_IMAGE_H
#define RAY_TRACER_IMAGE_H

typedef unsigned char byte;

struct Image {
    unsigned width;
    unsigned height;
    unsigned bytesPerPixel;

    byte *pixelData = nullptr;

    Image(unsigned int width, unsigned int height, byte *pixelData,
          unsigned int bytesPerPixel = 3);

    Image(Image const &image);

    Image(Image &&image) noexcept;

    ~Image();
};

#endif //RAY_TRACER_IMAGE_H
