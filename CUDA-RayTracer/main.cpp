#include <memory>
#include "frontends/image/ImageFrontend.h"
#include "backends/Backend.h"
#include "backends/solid_color_cuda/SolidColorCudaBackend.h"
#include "frontends/frontend_controller/FrontendController.h"
#include "frontends/sdl/SDLFrontend.h"

int main(int argc, char *argv[]) {
    std::vector<std::function<Frontend *()>> frontendConstructors;
    frontendConstructors.emplace_back([]() { return new SDLFrontend; });
    frontendConstructors.emplace_back([]() { return new ImageFrontend; });

    FrontendController frontendController(frontendConstructors);
    frontendController.waitForInit();

    std::unique_ptr<Backend> backend(new SolidColorCudaBackend);
    backend->setResolution(800, 600);
    frontendController.setImage(backend->render());

    frontendController.waitForTermination();

    return 0;
}
