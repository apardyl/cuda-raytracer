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

public:
    explicit FrontendController(
            std::vector<std::function<Frontend *()>> const &constructors);

    ~FrontendController();

    void waitForInit();

    void setImage(Image image);

    void waitForTermination();
};

#endif //RAY_TRACER_FRONTENDCONTROLLER_H
