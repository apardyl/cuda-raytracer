#ifndef RAY_TRACER_BACKEND_H
#define RAY_TRACER_BACKEND_H

typedef unsigned char byte;

class Backend {
protected:
    unsigned width = 0;
    unsigned height = 0;

public:
    virtual byte *render() = 0;

    virtual void setResolution(unsigned width, unsigned height);
};

#endif //RAY_TRACER_BACKEND_H
