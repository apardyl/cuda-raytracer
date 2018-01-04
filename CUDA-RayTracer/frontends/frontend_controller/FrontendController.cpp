#include <thread>
#include "FrontendController.h"

FrontendController::FrontendController(
        std::vector<std::function<Frontend *()>> const &constructors) :
        initLatch(static_cast<unsigned int>(constructors.size())),
        frontends(constructors.size(), nullptr) {
    frontendThreads.reserve(constructors.size());

    for (size_t i = 0; i < constructors.size(); ++i) {
        const auto &constructor = constructors[i];
        frontendThreads.emplace_back([&](int i, auto constructor) {
            std::unique_ptr<Frontend> frontend;
            {
                std::unique_lock<std::mutex> localLock(frontendsLock);
                frontend.reset(constructor());
                frontends[i] = frontend.get();
            }
            initLatch.countDown();

            frontend->run();

            {
                std::unique_lock<std::mutex> localLock(frontendsLock);
                frontends[i] = nullptr;
            }
        }, i, constructor);
    }
}

void FrontendController::waitForInit() {
    initLatch.await();
}

void FrontendController::setImage(Image image) {
    std::unique_lock<std::mutex> localLock(frontendsLock);

    for (auto const &frontend : frontends) {
        if (frontend != nullptr) {
            frontend->setImage(image);
        }
    }
}

void FrontendController::waitForTermination() {
    for (std::thread &thread : frontendThreads) {
        thread.join();
    }
}
