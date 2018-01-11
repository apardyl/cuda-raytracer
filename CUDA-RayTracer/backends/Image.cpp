#include <cstring>
#include "Image.h"

Image::Image(unsigned width, unsigned height) : width(width), height(height) {
    this->pixelData = new Color[sizeof(Color) * width * height];
}

Image::Image(unsigned int width, unsigned int height, Color *pixelData)
    : Image(width, height) {
    memcpy(this->pixelData, pixelData, sizeof(Color) * width * height);
}

Image::~Image() {
    if (pixelData != nullptr) {
        delete[] pixelData;
        pixelData = nullptr;
    }
}

Image::Image(Image const &image) :
    Image(image.width, image.height, image.pixelData) {
}

Image::Image(Image &&image) noexcept :
    width(image.width),
    height(image.height),
    pixelData(image.pixelData) {
    image.pixelData = nullptr;
}
