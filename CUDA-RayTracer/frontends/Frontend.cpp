#include "Frontend.h"

void Frontend::setBackendController(BackendController *backendController) {
    this->backendController = backendController;
}

bool Frontend::isDaemon() {
    return false;
}

void Frontend::terminate() {
    // do nothing
}
