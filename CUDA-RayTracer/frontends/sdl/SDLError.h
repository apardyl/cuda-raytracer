#ifndef RAY_TRACER_SDLERROR_H
#define RAY_TRACER_SDLERROR_H

#include <stdexcept>
#include <string>

class SDLError : public std::runtime_error {
public:
    explicit SDLError(std::string const &message);
};

#endif //RAY_TRACER_SDLERROR_H
