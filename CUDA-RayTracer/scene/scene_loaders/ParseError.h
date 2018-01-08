#ifndef RAY_TRACER_PARSEERROR_H
#define RAY_TRACER_PARSEERROR_H

#include <stdexcept>

class ParseError : public std::runtime_error {
public:
    explicit ParseError(const std::string &message);
};

#endif // RAY_TRACER_PARSEERROR_H
