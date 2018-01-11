#ifndef RAY_TRACER_RESOLUTION_H
#define RAY_TRACER_RESOLUTION_H

struct Resolution {
    unsigned const int width = 0;
    unsigned const int height = 0;

    Resolution(unsigned int width, unsigned int height);
};

#endif //RAY_TRACER_RESOLUTION_H
