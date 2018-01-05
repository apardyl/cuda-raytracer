#if defined(__STDC_LIB_EXT1__) || defined(_MSC_VER)
#define __STDC_WANT_LIB_EXT1__ 1
#else
#define fopen_s(pFile, filename, mode) \
    ((*(pFile)) = fopen((filename), (mode))) == NULL
#endif

#include <png.h>
#include <cstdio>
#include "ImageFrontend.h"
#include "ImageError.h"

ImageFrontend::ImageFrontend(std::string const &filename) :
        filename(filename) {
}

ImageFrontend::~ImageFrontend() = default;

void ImageFrontend::run() {
    while (!shouldTerminate || image != nullptr) {
        std::unique_lock<std::mutex> localLock(lock);
        if (image != nullptr) {
            savePNG(*image);
            image = nullptr;
        }

        if (shouldTerminate) {
            break;
        }

        condition.wait(localLock);
    }
}

void ImageFrontend::setImage(Image image) {
    std::unique_lock<std::mutex> localLock(lock);
    this->image = std::make_unique<Image>(std::move(image));
    condition.notify_all();
}

bool ImageFrontend::isDaemon() {
    return true;
}

void ImageFrontend::terminate() {
    std::unique_lock<std::mutex> localLock(lock);
    shouldTerminate = true;
    condition.notify_all();
}

void ImageFrontend::savePNG(Image const &image) {
    FILE *fp = nullptr;
    if (fopen_s(&fp, filename.c_str(), "wb") || fp == nullptr) {
        throw ImageError("Could not open file for writing");
    }

    // Init PNG structure
    png_structp pngStruct = png_create_write_struct(
            PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (pngStruct == nullptr) {
        throw ImageError("Could not create PNG structure");
    }

    // Init info structure
    png_infop pngInfo = png_create_info_struct(pngStruct);
    if (pngInfo == nullptr) {
        throw ImageError("Could not create info structure");
    }

    // IO
    png_init_io(pngStruct, fp);

    // Write header
    png_set_IHDR(pngStruct, pngInfo, image.width, image.height,
                 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(pngStruct, pngInfo);

    // Rewrite image
    auto row = new png_byte[image.width * image.bytesPerPixel];
    for (unsigned y = 0; y < image.height; ++y) {
        for (unsigned x = 0; x < image.width; ++x) {
            unsigned n = image.bytesPerPixel;
            for (unsigned i = 0; i < n; ++i) {
                row[x * n + i] = image.pixelData[(y * image.width + x) * n + i];
            }
        }
        png_write_row(pngStruct, row);
    }

    // Cleanup
    png_write_end(pngStruct, nullptr);

    fclose(fp);
    delete[] row;
    png_free_data(pngStruct, pngInfo, PNG_FREE_ALL, -1);
    png_destroy_write_struct(&pngStruct, nullptr);
}
