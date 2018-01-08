#ifndef RAY_TRACER_BACKENDCONTROLLER_H
#define RAY_TRACER_BACKENDCONTROLLER_H

#include <mutex>
#include <condition_variable>

#include "backends/Backend.h"

/**
 * Thread-safe class that a frontend can use to control the backend: set
 * resolution, rendering options, re-render the image, etc.
 */
class BackendController {
private:
    std::recursive_mutex *lock;
    std::condition_variable_any *condition;

    bool refreshRequested = false;
    bool resolutionChangeRequested = false;
    unsigned width;
    unsigned height;

public:
    BackendController(std::recursive_mutex *lock,
                      std::condition_variable_any *condition);

    void refresh();

    void setResolution(unsigned width, unsigned height);

    bool isRefreshRequested() const;

    void applyBackendSettings(Backend *backend);
};

#endif //RAY_TRACER_BACKENDCONTROLLER_H
