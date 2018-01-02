#ifndef RAY_TRACER_SDLFRONTEND_H
#define RAY_TRACER_SDLFRONTEND_H

#include <SDL.h>

class SDLFrontend {
private:
    const int SCREEN_WIDTH = 800;
    const int SCREEN_HEIGHT = 600;
    const int FPS_CAP = 30;
    const int SCREEN_TICKS_PER_FRAME = 1000 / FPS_CAP;

    SDL_Window *window = nullptr;
    SDL_Surface *screenSurface = nullptr;
    SDL_Surface *renderedImage = nullptr;

    void render();

    void createRenderedImageSurface();

public:
    SDLFrontend();

    ~SDLFrontend();

    void run();
};

#endif //RAY_TRACER_SDLFRONTEND_H
