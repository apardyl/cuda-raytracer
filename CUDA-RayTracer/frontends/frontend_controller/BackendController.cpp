#include <backends/Backend.h>
#include "BackendController.h"


BackendController::BackendController(
        std::recursive_mutex *lock, std::condition_variable_any *condition) :
        lock(lock), condition(condition) {}

void BackendController::refresh() {
    std::unique_lock<std::recursive_mutex> localLock(*lock);
    refreshRequested = true;
    condition->notify_all();
}

void BackendController::setResolution(unsigned width, unsigned height) {
    std::unique_lock<std::recursive_mutex> localLock(*lock);
    this->width = width;
    this->height = height;
    resolutionChangeRequested = true;
}

bool BackendController::isRefreshRequested() const {
    std::unique_lock<std::recursive_mutex> localLock(*lock);
    return refreshRequested;
}

void BackendController::applyBackendSettings(Backend *backend) {
    std::unique_lock<std::recursive_mutex> localLock(*lock);

    if (resolutionChangeRequested) {
        backend->setResolution(width, height);
        resolutionChangeRequested = false;
    }
    refreshRequested = false;
}
