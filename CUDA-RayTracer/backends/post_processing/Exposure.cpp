#include "Exposure.h"
#include <cmath>
#include <algorithm>

void Exposure::autoExposure(const Image *image) {
    float max = 0;
#pragma omp parallel for
    for (int i = 0; i < image->height; i++) {
        for (unsigned j = 0; j < image->width; j++) {
            const int id = i * image->width + j;
            max = std::max(
                max, 0.2126f * image->pixelData[id].red + 0.715160f * image->pixelData[id].green +
                0.072169f * image->pixelData[id].blue);
        }
    }
    exposure = std::min(1.0f / max - 0.1f, 1.0f);
}

Exposure::Exposure(float exposure): exposure(exposure) {
}

std::unique_ptr<Image> Exposure::process(std::unique_ptr<Image> image) {
    if (exposure == 0) {
        autoExposure(image.get());
    }
#pragma omp parallel for
    for (int i = 0; i < image->height; i++) {
        for (unsigned j = 0; j < image->width; j++) {
            const int id = i * image->width + j;
            image->pixelData[id] = image->pixelData[id] * exposure;
        }
    }
    return image;
}
