#include "UnknownFormatException.h"

UnknowFormatException::UnknowFormatException(const std::string &message): runtime_error(message) {
}
