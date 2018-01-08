#ifndef RAY_TRACER_BACKENDINITIALIZER_H
#define RAY_TRACER_BACKENDINITIALIZER_H

#include <string>
#include <vector>

#include "backends/Backend.h"

class BackendInitializer {
#ifdef ENABLE_CUDA
    static const std::string SOLID_COLOR_CUDA_NAME;
    static const std::string RANDOM_CUDA_NAME;
#endif
    static const std::string RAY_TRACER_OPENMP;
    static const std::vector<std::string> BACKEND_LIST;

public:
    static Backend *createBackend(std::string const &name);

    static std::string generateBackendListString();

    static bool isBackendNameValid(std::string const &backendName);
};

#endif //RAY_TRACER_BACKENDINITIALIZER_H
