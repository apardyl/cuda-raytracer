#ifndef RAY_TRACER_SDLFRONTEND_H
#define RAY_TRACER_SDLFRONTEND_H

#include <SDL.h>
#include <string>

class SDLFrontend {
private:
    SDL_Window *window = nullptr;
    SDL_Surface *screenSurface = nullptr;

    const int SCREEN_WIDTH = 800;
    const int SCREEN_HEIGHT = 600;
    const int FPS_CAP = 30;
    const int SCREEN_TICKS_PER_FRAME = 1000 / FPS_CAP;
public:
    SDLFrontend();

    ~SDLFrontend();

    void run();

    void render();
};


#endif //RAY_TRACER_SDLFRONTEND_H
