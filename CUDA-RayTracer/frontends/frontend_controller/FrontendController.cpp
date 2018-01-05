#include <thread>
#include "frontends/sdl/SDLFrontend.h"
#include "FrontendController.h"

FrontendController::FrontendController(
        std::vector<std::function<Frontend *()>> const &constructors) :
        initLatch(static_cast<unsigned int>(constructors.size())) {
    frontends.reserve(constructors.size());

    for (size_t i = 0; i < constructors.size(); ++i) {
        const auto &constructor = constructors[i];
        frontends.emplace_back(std::make_unique<std::thread>(
                std::thread([&](int i, auto constructor) {
                    std::shared_ptr<Frontend> frontend(constructor());
                    frontends[i].setFrontend(frontend);
                    initLatch.countDown();

                    frontend->run();
                }, i, constructor)));
    }
}

FrontendController::~FrontendController() {
    for (auto &frontend : frontends) {
        const std::shared_ptr<Frontend> &frontendPtr = frontend.frontend.lock();
        if (frontendPtr) {
            frontendPtr->terminate();
        }
    }

    for (auto &frontend : frontends) {
        if (frontend.thread->joinable()) {
            frontend.thread->join();
        }
    }
}

void FrontendController::waitForInit() {
    initLatch.await();
}

void FrontendController::setImage(Image image) {
    for (auto const &frontend : frontends) {
        const std::shared_ptr<Frontend> &frontendPtr = frontend.frontend.lock();
        if (frontendPtr) {
            frontendPtr->setImage(image);
        }
    }
}

void FrontendController::waitForTermination() {
    for (auto &frontend : frontends) {
        if (!frontend.isDaemon && frontend.thread->joinable()) {
            frontend.thread->join();
        }
    }
}
