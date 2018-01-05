#include "ImageError.h"

ImageError::ImageError(std::string const &message) :
        runtime_error("Image saving error: " + message) {
}
