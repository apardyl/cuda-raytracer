#ifndef RAY_TRACER_FRONTEND_H
#define RAY_TRACER_FRONTEND_H

#include <backends/Image.h>

class Frontend {
public:
    virtual ~Frontend() = default;

    virtual void run() = 0;

    virtual void setImage(Image image) = 0;

    /**
     * @return whether the frontend is a daemon (i.e. should be terminated
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
