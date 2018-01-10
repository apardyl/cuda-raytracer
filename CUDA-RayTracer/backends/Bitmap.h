#ifndef RAY_TRACER_BITMAP_H
#define RAY_TRACER_BITMAP_H

struct Image;
typedef unsigned char byte;

struct Bitmap {
    unsigned width;
    unsigned height;
    unsigned bytesPerPixel;

    byte *pixelData = nullptr;

    explicit Bitmap(const Image &image);

    Bitmap(unsigned int width, unsigned int height, unsigned int bytesPerPixel = 3);

    Bitmap(unsigned int width, unsigned int height, byte *pixelData,
           unsigned int bytesPerPixel = 3);

    Bitmap(Bitmap const &bitmap);

    Bitmap(Bitmap &&bitmap) noexcept;

    ~Bitmap();
};

#endif //RAY_TRACER_BITMAP_H
