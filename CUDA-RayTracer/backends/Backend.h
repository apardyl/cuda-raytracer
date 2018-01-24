#ifndef RAY_TRACER_BACKEND_H
#define RAY_TRACER_BACKEND_H

#include "Image.h"
#include "scene/Scene.h"
#include <memory>

class Backend {
protected:
    unsigned width = 0;
    unsigned height = 0;
    std::unique_ptr<Scene> scene;

public:
    virtual ~Backend() = default;

    virtual Image render() = 0;

    virtual void setResolution(unsigned width, unsigned height);

    virtual void setScene(std::unique_ptr<Scene> scene);
};

#endif //RAY_TRACER_BACKEND_H
