#ifndef RAY_TRACER_UNKNOWFORMATEXCEPTION_H
#define RAY_TRACER_UNKNOWFORMATEXCEPTION_H

#include <stdexcept>

class UnknowFormatException : public std::runtime_error {
public:
    explicit UnknowFormatException(const std::string &message);
};

#endif // RAY_TRACER_UNKNOWFORMATEXCEPTION_H
