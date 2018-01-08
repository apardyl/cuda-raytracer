#ifndef RAY_TRACER_APPLICATIONOPTIONS_H
#define RAY_TRACER_APPLICATIONOPTIONS_H

#include <string>
#include "CompileSettings.h"

struct ApplicationOptions {
    unsigned width;
    unsigned height;

    std::string backendName;

    bool sdlFrontendEnabled;
#if GTK_ENABLED
    bool gtkFrontendEnabled;
#endif
    bool imageFrontendEnabled;
    std::string outputFilename;

    bool sceneLoaderEnabled;
    std::string sceneFilename;
};

#endif //RAY_TRACER_APPLICATIONOPTIONS_H
