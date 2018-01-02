#include "SDLFrontend.h"
#include "SDLError.h"

SDLFrontend::SDLFrontend() {
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
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT) {
                quit = true;
            }
        }

        SDL_FillRect(
                screenSurface, nullptr,
                SDL_MapRGB(screenSurface->format, 0xFF, 0xFF, 0xFF));
        SDL_UpdateWindowSurface(window);
    }
}

