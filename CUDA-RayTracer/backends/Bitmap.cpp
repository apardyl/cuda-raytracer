#include <cstring>
#include "Bitmap.h"
#include "Image.h"
#include <algorithm>
#include <iostream>

#define BYTES_PER_PIXEL 3

Bitmap::Bitmap(const Image &image) : Bitmap(image.width, image.height, BYTES_PER_PIXEL) {
#pragma omp parallel for
    for (int i = 0; i < image.height; i++) {
        for (unsigned j = 0; j < image.width; j++) {
            const unsigned id = i * image.width + j;
            pixelData[id * BYTES_PER_PIXEL] = std::min(image.pixelData[id].red, static_cast<float>(1)) * 255;
            pixelData[id * BYTES_PER_PIXEL + 1] = std::min(image.pixelData[id].green, static_cast<float>(1)) * 255;
            pixelData[id * BYTES_PER_PIXEL + 2] = std::min(image.pixelData[id].blue, static_cast<float>(1)) * 255;
        }
    }
}

Bitmap::Bitmap(unsigned width, unsigned height, unsigned bytesPerPixel)
    : width(width), height(height), bytesPerPixel(bytesPerPixel) {
    this->pixelData = new byte[sizeof(byte) * bytesPerPixel * width * height];
}

Bitmap::Bitmap(unsigned int width, unsigned int height, byte *pixelData,
               unsigned int bytesPerPixel) : Bitmap(width, height, bytesPerPixel) {
    memcpy(this->pixelData, pixelData, sizeof(byte) * bytesPerPixel * width * height);
}


Bitmap::~Bitmap() {
    if (pixelData != nullptr) {
        delete[] pixelData;
        pixelData = nullptr;
    }
}

Bitmap::Bitmap(Bitmap const &bitmap) :
    Bitmap(bitmap.width, bitmap.height, bitmap.pixelData, bitmap.bytesPerPixel) {
}

Bitmap::Bitmap(Bitmap &&bitmap) noexcept :
    width(bitmap.width),
    height(bitmap.height),
    bytesPerPixel(bitmap.bytesPerPixel),
    pixelData(bitmap.pixelData) {
    bitmap.pixelData = nullptr;
}
