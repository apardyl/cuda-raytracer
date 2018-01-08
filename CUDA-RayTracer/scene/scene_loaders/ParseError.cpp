#include "ParseError.h"

ParseError::ParseError(const std::string &message) : runtime_error(message) {
}
