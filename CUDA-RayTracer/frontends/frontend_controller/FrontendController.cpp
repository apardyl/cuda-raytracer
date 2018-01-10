#include <thread>
#include "frontends/sdl/SDLFrontend.h"
#include "FrontendController.h"

FrontendController::FrontendController(
        std::vector<std::function<Frontend *()>> const &constructors,
        BackendController &backendController) :
        frontendsTerminated(false),
        initLatch(static_cast<unsigned int>(constructors.size())) {
    frontends.resize(constructors.size());

    for (size_t i = 0; i < constructors.size(); ++i) {
        frontends[i].thread = std::make_unique<std::thread>(
                std::thread([&](int i, auto constructor,
                                auto *backendController) {
                    std::shared_ptr<Frontend> frontend(constructor());
                    frontend->setBackendController(backendController);
                    frontends[i].setFrontend(frontend);
                    initLatch.countDown();

                    frontend->run();
                }, i, constructors[i], &backendController));
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

void FrontendController::setImage(Bitmap image) {
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
    frontendsTerminated = true;
}

bool FrontendController::areFrontendsTerminated() const {
    return static_cast<bool>(frontendsTerminated);
}
