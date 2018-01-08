#include "Backend.h"

void Backend::setResolution(unsigned width, unsigned height) {
    this->width = width;
    this->height = height;
}

void Backend::setScene(std::unique_ptr<Scene> scene) {
    this->scene = std::move(scene);
}
