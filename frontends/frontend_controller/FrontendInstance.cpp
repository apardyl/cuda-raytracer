#include "FrontendInstance.h"

FrontendInstance::FrontendInstance(std::unique_ptr<std::thread> thread) :
        thread(std::move(thread)) {}

void FrontendInstance::setFrontend(std::shared_ptr<Frontend> const &frontend) {
    this->isDaemon = frontend->isDaemon();
    this->frontend = frontend;
}
