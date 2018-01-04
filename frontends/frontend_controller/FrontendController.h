#ifndef RAY_TRACER_FRONTENDCONTROLLER_H
#define RAY_TRACER_FRONTENDCONTROLLER_H

#include <vector>
#include <functional>
#include <thread>
#include "frontends/Frontend.h"
#include "CountDownLatch.h"

class FrontendController {
private:
    CountDownLatch initLatch;
    std::vector<Frontend *> frontends;
    std::vector<std::thread> frontendThreads;
    std::mutex frontendsLock;

public:
    explicit FrontendController(
            std::vector<std::function<Frontend *()>> const &constructors);

    void waitForInit();

    void setImage(Image image);

    void waitForTermination();
};

#endif //RAY_TRACER_FRONTENDCONTROLLER_H
