#include "BackendInitializer.h"

#include <numeric>
#include <algorithm>

using namespace std::string_literals;

const std::string BackendInitializer::SOLID_COLOR_CUDA_NAME = "SolidColorCuda";
const std::string BackendInitializer::RAY_TRACER_OPENMP = "RayTracerOpenMP";
const std::array<std::string, 2> BackendInitializer::BACKEND_LIST = {
        SOLID_COLOR_CUDA_NAME, RAY_TRACER_OPENMP};

Backend *BackendInitializer::createBackend(std::string const &name) {
    if (name == SOLID_COLOR_CUDA_NAME) {
        return new SolidColorCudaBackend;
    } else if (name == RAY_TRACER_OPENMP) {
            return new RayTracingOpenMP;
    }
    return nullptr;
}

std::string BackendInitializer::generateBackendListString() {
    return std::accumulate(
            BACKEND_LIST.begin(), BACKEND_LIST.end(), ""s,
            [](std::string const &s1, std::string const &s2) {
                return s1.empty() ? s2 : s1 + ", " + s2;
            });
}

bool BackendInitializer::isBackendNameValid(std::string const &backendName) {
    return std::find(BACKEND_LIST.begin(), BACKEND_LIST.end(), backendName) !=
           BACKEND_LIST.end();
}
