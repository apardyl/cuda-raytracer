#include "Application.h"

#include <iostream>

#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/parsers.hpp>

#include "CompileSettings.h"
#include "BackendInitializer.h"
#include "frontends/sdl/SDLFrontend.h"
#include "frontends/gtk/GTKFrontend.h"
#include "frontends/image/ImageFrontend.h"
#include "scene/SceneLoader.h"
#include "scene/scene_loaders/IOException.h"
#include "scene/scene_loaders/ParseError.h"
#include "scene/scene_loaders/UnknownFormatException.h"
#include "backends/Bitmap.h"

using namespace std::string_literals;

Application::Application(int argc, char **argv) :
        argc(argc), argv(argv) {
    run();
}

void Application::parseCommandLine() {
    po::options_description all("Allowed options");
    all
            .add(createRenderingOptions())
            .add(createFrontendsOptions())
            .add(createSceneOptions())
            .add(createGeneralOptions());

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, all), vm);
        po::notify(vm);
    } catch (po::error &e) {
        std::cout << "Error: " << e.what() << std::endl;
        exit(1);
    }

    if (vm.count("help")) {
        std::cout << "Usage: " << argv[0] << " [options]"
                  << std::endl
                  << std::endl
                  << all
                  << std::endl;
        exit(0);
    }

    rewriteOptions(vm);
}

void Application::rewriteOptions(po::variables_map const &vm) {
    options.width = vm["width"].as<unsigned>();
    options.height = vm["height"].as<unsigned>();

    options.backendName = vm["backend"].as<std::string>();

    options.imageFrontendEnabled = vm.count("output") > 0;
    if (options.imageFrontendEnabled) {
        options.outputFilename = vm["output"].as<std::string>();
    }
    options.sdlFrontendEnabled = vm.count("sdl") > 0;
#if GTK_ENABLED
    options.gtkFrontendEnabled = vm.count("gtk") > 0;
#endif
    options.sceneLoaderEnabled = vm.count("input-file") > 0;
    if(options.sceneLoaderEnabled) {
        options.sceneFilename = vm["input-file"].as<std::string>();
    }
}

po::options_description Application::createGeneralOptions() {
    po::options_description general("General");

    general.add_options()
            ("help", "display this help and exit");

    return general;
}

void validateBackend(std::string const &backendName) {
    if (!BackendInitializer::isBackendNameValid(backendName)) {
        throw po::validation_error(po::validation_error::invalid_option_value,
                                   "backend", backendName);
    }
}

po::options_description Application::createRenderingOptions() {
    po::options_description rendering("Rendering");

    std::string backendDescription =
            "set the backend used (available: "s +
            BackendInitializer::generateBackendListString() + ")";
    rendering.add_options()
            ("width,w",
             boost::program_options::value<unsigned>()
                     ->default_value(800)
                     ->value_name("WIDTH"),
             "set rendered image width")
            ("height,h",
             boost::program_options::value<unsigned>()
                     ->default_value(600)
                     ->value_name("HEIGHT"),
             "set rendered image height")
            ("backend,b",
             boost::program_options::value<std::string>()
                     ->default_value("RayTracingOpenMP")
                     ->value_name("BACKEND_NAME")
                     ->notifier(&validateBackend),
             backendDescription.c_str());

    return rendering;
}

po::options_description Application::createFrontendsOptions() {
    po::options_description frontends("Frontends");

    std::string sdlDescription = "run SDL2 frontend";
    std::string gtkDescription = "run GTK+ 3 frontend";
    std::string defaultDescription = " (default if no other is specified)";
#if GTK_ENABLED
    gtkDescription += defaultDescription;
#else
    sdlDescription += defaultDescription;
#endif

    frontends.add_options()
            ("sdl", sdlDescription.c_str())
#if GTK_ENABLED
            ("gtk", gtkDescription.c_str())
#endif
            ("output,o",
             po::value<std::string>()->value_name("OUTPUT_FILE_NAME"),
             "set output image file name");

    return frontends;
}

po::options_description Application::createSceneOptions() {
    po::options_description scene("Scene");

    scene.add_options()("input-file,f",
                        boost::program_options::value<std::string>()
                        ->value_name("SCENE_FILE_NAME"),
                        "load scene from file");

    return scene;
}

void Application::run() {
    parseCommandLine();

    BackendController backendController(
            &executionLock, &executionCondition);
    FrontendController frontendController(
            createFrontendList(), backendController);
    frontendController.waitForInit();

    std::unique_ptr<Backend> backend(
            BackendInitializer::createBackend(options.backendName));

    if (options.sceneLoaderEnabled) {
        try {
            std::unique_ptr<Scene> scene =
                std::make_unique<Scene>(SceneLoader::loadFromFile(options.sceneFilename));
            backend->setScene(std::move(scene));
        } catch(const std::runtime_error& exception) {
            std::cerr << "[ERROR] " << exception.what() << std::endl;
            exit(1);
        }
    }

    backend->setResolution(options.width, options.height);
    Bitmap bitmap(backend->render());
    frontendController.setImage(bitmap);

    std::thread frontendControllerThread([&]() {
        frontendController.waitForTermination();
        {
            std::unique_lock<std::recursive_mutex> localLock(executionLock);
            executionCondition.notify_all();
        }
    });

    while (!frontendController.areFrontendsTerminated()) {
        // Wait either for frontends termination, or re-render request

        std::unique_lock<std::recursive_mutex> localLock(executionLock);
        if (backendController.isRefreshRequested()) {
            backendController.applyBackendSettings(backend.get());
            Bitmap bitmap(backend->render());
            frontendController.setImage(bitmap);
        }

        executionCondition.wait(localLock);
    }

    if (frontendControllerThread.joinable()) {
        frontendControllerThread.join();
    }
}

std::vector<std::function<Frontend *()>>
Application::createFrontendList() const {
    std::vector<std::function<Frontend *()>> frontendConstructors;

    bool createdFrontend = false;

    if (options.imageFrontendEnabled) {
        frontendConstructors.emplace_back([&]() {
            return new ImageFrontend(options.outputFilename);
        });
        createdFrontend = true;
    }

    // SDL is the default frontend: created if no other was specified
    if (options.sdlFrontendEnabled || (!createdFrontend && !GTK_ENABLED)) {
        frontendConstructors.emplace_back([]() { return new SDLFrontend; });
    }

#if GTK_ENABLED
    if (options.gtkFrontendEnabled || !createdFrontend) {
        frontendConstructors.emplace_back([&]() {
            return new GTKFrontend(options);
        });
        createdFrontend = true;
    }
#endif

    return frontendConstructors;
}
