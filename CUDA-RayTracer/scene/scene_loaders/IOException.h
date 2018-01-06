#ifndef RAY_TRACER_IOEXCEPTION_H
#define RAY_TRACER_IOEXCEPTION_H

#include <stdexcept>

class IOException : public std::runtime_error {
public:
    explicit IOException(const std::string &message);
};

#endif // RAY_TRACER_IOEXCEPTION_H
