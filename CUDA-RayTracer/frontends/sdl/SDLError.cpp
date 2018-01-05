#include "SDLError.h"
#include <SDL.h>

SDLError::SDLError(std::string const &message)
        : runtime_error(message + ": " + SDL_GetError()) {
}
