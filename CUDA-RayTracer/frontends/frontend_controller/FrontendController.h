#ifndef RAY_TRACER_FRONTENDCONTROLLER_H
#define RAY_TRACER_FRONTENDCONTROLLER_H

#include <vector>
#include <functional>
#include <thread>
#include "frontends/Frontend.h"
#include "CountDownLatch.h"
#include "FrontendInstance.h"

class FrontendController {
private:
    CountDownLatch initLatch;
    std::vector<FrontendInstance> frontends;
    std::atomic<bool> frontendsTerminated;

public:
    explicit FrontendController(
            std::vector<std::function<Frontend *()>> const &constructors,
            BackendController &backendController);

    ~FrontendController();

    void waitForInit();

    void setImage(const Bitmap& image);

    void waitForTermination();

    bool areFrontendsTerminated() const;
};

#endif //RAY_TRACER_FRONTENDCONTROLLER_H
