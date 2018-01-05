#include "SDLFrontend.h"
#include "SDLError.h"

SDLFrontend::SDLFrontend() {
    renderedImage = nullptr;

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        throw SDLError("SDL could not initialize");
    }

    window = SDL_CreateWindow(
            "Ray Tracer",
            SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
            SCREEN_WIDTH, SCREEN_HEIGHT,
            SDL_WINDOW_SHOWN);
    if (window == nullptr) {
        throw SDLError("Window could not be created");
    }

    screenSurface = SDL_GetWindowSurface(window);
}

SDLFrontend::~SDLFrontend() {
    SDL_DestroyWindow(window);
    window = nullptr;
    SDL_Quit();
}

void SDLFrontend::run() {
    bool quit = false;
    SDL_Event e{};

    while (!quit) {
        Uint32 startTicks = SDL_GetTicks();

        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT) {
                quit = true;
            }
        }

        render();

        Uint32 frameTicks = SDL_GetTicks() - startTicks;
        if (frameTicks < SCREEN_TICKS_PER_FRAME) {
            SDL_Delay(SCREEN_TICKS_PER_FRAME - frameTicks);
        }
    }
}

void SDLFrontend::render() {
    SDL_FillRect(screenSurface, nullptr, 0);
    if (static_cast<SDL_Surface*>(renderedImage) != nullptr) {
        SDL_BlitSurface(renderedImage, nullptr, screenSurface, nullptr);
        SDL_UpdateWindowSurface(window);
    }
}

void SDLFrontend::setImage(Image image) {
    SDL_Surface * surface = SDL_CreateRGBSurfaceFrom(
            image.pixelData, image.width, image.height, 24,
            image.width * image.bytesPerPixel,
            0x0000ff, 0x00ff00, 0xff00000, 0);

    if (surface == nullptr) {
        throw SDLError("Could not create rendered image surface");
    }

    renderedImage = SDL_ConvertSurface(surface, screenSurface->format, 0);
    if (static_cast<SDL_Surface*>(renderedImage) == nullptr) {
        throw SDLError("Could not convert rendered image surface");
    }
}
