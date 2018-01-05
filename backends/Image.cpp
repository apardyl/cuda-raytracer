#include <cstring>
#include "Image.h"

Image::Image(unsigned int width, unsigned int height, byte *pixelData,
             unsigned int bytesPerPixel)
        : width(width), height(height), bytesPerPixel(bytesPerPixel) {
    size_t numBytes =
            sizeof(byte) * bytesPerPixel * width * height;
    this->pixelData = new byte[numBytes];
    memcpy(this->pixelData, pixelData, numBytes);
}

Image::~Image() {
    if (pixelData != nullptr) {
        delete[] pixelData;
        pixelData = nullptr;
    }
}

Image::Image(Image const &image) :
        Image(image.width, image.height, image.pixelData, image.bytesPerPixel) {
}

Image::Image(Image &&image) noexcept :
        width(image.width),
        height(image.height),
        bytesPerPixel(image.bytesPerPixel),
        pixelData(image.pixelData) {
    image.pixelData = nullptr;
}
