#ifndef RAY_TRACER_IMAGEERROR_H
#define RAY_TRACER_IMAGEERROR_H

#include <stdexcept>

class ImageError : public std::runtime_error{
public:
    explicit ImageError(std::string const &message);
};

#endif //RAY_TRACER_IMAGEERROR_H
