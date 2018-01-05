#ifndef RAY_TRACER_FRONTENDINSTANCE_H
#define RAY_TRACER_FRONTENDINSTANCE_H

#include <thread>
#include "frontends/Frontend.h"

struct FrontendInstance {
    std::weak_ptr<Frontend> frontend;
    std::unique_ptr<std::thread> thread;
    bool isDaemon;

    explicit FrontendInstance(std::unique_ptr<std::thread> thread);

    void setFrontend(std::shared_ptr<Frontend> const &frontend);
};

#endif //RAY_TRACER_FRONTENDINSTANCE_H
