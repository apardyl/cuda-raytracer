#ifndef RAY_TRACER_BACKEND_H
#define RAY_TRACER_BACKEND_H

#include "Image.h"

class Backend {
protected:
    unsigned width = 0;
    unsigned height = 0;

public:
    virtual ~Backend() = default;

    virtual Image render() = 0;

    virtual void setResolution(unsigned width, unsigned height);
};

#endif //RAY_TRACER_BACKEND_H
