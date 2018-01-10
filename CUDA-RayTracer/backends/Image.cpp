#include <cstring>
#include "Image.h"

Image::Image(unsigned int width, unsigned int height, Color *pixelData)
    : width(width), height(height) {
    const size_t numBytes =
        sizeof(Color) * width * height;
    this->pixelData = new Color[numBytes];
    memcpy(this->pixelData, pixelData, numBytes);
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
