#include "IOException.h"

IOException::IOException(const std::string &message) : runtime_error(message) {
}
