#ifndef RAY_TRACER_FRONTEND_H
#define RAY_TRACER_FRONTEND_H

#include "frontends/frontend_controller/BackendController.h"
#include "backends/Bitmap.h"

class Frontend {
protected:
    BackendController *backendController = nullptr;

public:
    virtual ~Frontend() = default;

    virtual void run() = 0;

    virtual void setImage(Bitmap image) = 0;

    virtual void setBackendController(BackendController *backendController);

    /**
     * Return whether the frontend is a daemon (i.e. should be terminated
     * after all other frontends have been terminated).
     */
    virtual bool isDaemon();

    /**
     * Terminate the frontend. Useful mainly for the daemon frontends to allow
     * them to terminate gracefully. Default implementation does nothing.
     */
    virtual void terminate();
};

#endif //RAY_TRACER_FRONTEND_H
