#include "FrontendInstance.h"

void FrontendInstance::setFrontend(std::shared_ptr<Frontend> const &frontend) {
    this->isDaemon = frontend->isDaemon();
    this->frontend = frontend;
}
