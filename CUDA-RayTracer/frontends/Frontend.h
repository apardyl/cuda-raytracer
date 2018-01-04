#ifndef RAY_TRACER_FRONTEND_H
#define RAY_TRACER_FRONTEND_H

#include <backends/Image.h>

class Frontend {
public:
    virtual ~Frontend() = default;

    virtual void run() = 0;

    virtual void setImage(Image image) = 0;
};

#endif //RAY_TRACER_FRONTEND_H
