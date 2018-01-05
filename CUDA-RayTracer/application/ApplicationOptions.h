#ifndef RAY_TRACER_APPLICATIONOPTIONS_H
#define RAY_TRACER_APPLICATIONOPTIONS_H

#include <string>

struct ApplicationOptions {
    unsigned width;
    unsigned height;

    std::string backendName;

    bool sdlFrontendEnabled;
    bool imageFrontendEnabled;
    std::string outputFilename;
};

#endif //RAY_TRACER_APPLICATIONOPTIONS_H
