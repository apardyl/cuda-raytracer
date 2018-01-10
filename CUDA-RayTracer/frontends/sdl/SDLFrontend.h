#ifndef RAY_TRACER_SDLFRONTEND_H
#define RAY_TRACER_SDLFRONTEND_H

#include <SDL.h>
#include <atomic>
#include "frontends/Frontend.h"
#include "backends/Bitmap.h"

class SDLFrontend : public Frontend {
private:
    const unsigned SCREEN_WIDTH = 800;
    const unsigned SCREEN_HEIGHT = 600;
    const unsigned FPS_CAP = 30;
    const unsigned SCREEN_TICKS_PER_FRAME = 1000 / FPS_CAP;

    SDL_Window *window = nullptr;
    SDL_Surface *screenSurface = nullptr;
    std::atomic<SDL_Surface *> renderedImage;

    void render();

public:
    SDLFrontend();

    ~SDLFrontend() override;

    void run() override;

    void setImage(Bitmap image) override;
};

#endif //RAY_TRACER_SDLFRONTEND_H
