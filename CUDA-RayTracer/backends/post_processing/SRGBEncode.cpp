#include "SRGBEncode.h"

float SRGBEncode::encodeSrgb(float f) {
    if (f <= 0.0032308) {
        return 12.92 * f;
    } else {
        return 1.055 * powf(f, 0.4166667f) - 0.055f;
    }
}

std::unique_ptr<Image> SRGBEncode::process(std::unique_ptr<Image> image) {
#pragma omp parallel for
    for(int i = 0; i < image->height; i++) {
        for(unsigned j = 0; j < image->width; j++) {
            const int id = i * image->width + j;
            image->pixelData[id].red = encodeSrgb(image->pixelData[id].red);
            image->pixelData[id].green = encodeSrgb(image->pixelData[id].green);
            image->pixelData[id].blue = encodeSrgb(image->pixelData[id].blue);
        }
    }
    return image;
}
