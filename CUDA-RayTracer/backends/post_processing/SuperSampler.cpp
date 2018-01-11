#include "SuperSampler.h"

#define ID(y, x) ((y) * image->width + x)

std::unique_ptr<Image> SuperSampler::process(std::unique_ptr<Image> image) {
    std::unique_ptr<Image> out =
        std::make_unique<Image>(Image(image->width / 2, image->height / 2));

#pragma omp parallel for
    for (int i = 0; i < out->height; i++) {
        for (unsigned j = 0; j < out->width; j++) {
            out->pixelData[i * out->width + j] = (image->pixelData[ID(i * 2, j * 2)]
                + image->pixelData[ID(i * 2 + 1, j * 2)]
                + image->pixelData[ID(i * 2, j * 2 + 1)]
                + image->pixelData[ID(i * 2 + 1, j * 2 + 1)]) / 4;
        }
    }

    return out;
}
