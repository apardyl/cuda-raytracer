cmake_minimum_required(VERSION 3.9)
project(ray_tracer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${ray_tracer_SOURCE_DIR}/cmake")

# CUDA
find_package(CUDA REQUIRED)

# Project files
include_directories(${ray_tracer_SOURCE_DIR})

cuda_add_executable(
        ray_tracer
        main.cpp
        backends/Image.cpp backends/Image.h
        frontends/Frontend.cpp frontends/Frontend.h
        frontends/frontend_controller/FrontendInstance.cpp frontends/frontend_controller/FrontendInstance.h
        frontends/frontend_controller/CountDownLatch.cpp frontends/frontend_controller/CountDownLatch.h
        frontends/frontend_controller/FrontendController.cpp frontends/frontend_controller/FrontendController.h
        frontends/sdl/SDLFrontend.cpp frontends/sdl/SDLFrontend.h
        frontends/sdl/SDLError.cpp frontends/sdl/SDLError.h
        frontends/image/ImageError.cpp frontends/image/ImageError.h
        frontends/image/ImageFrontend.cpp frontends/image/ImageFrontend.h
        backends/Backend.cpp backends/Backend.h
        backends/solid_color_cuda/SolidColorCudaBackend.cpp backends/solid_color_cuda/SolidColorCudaBackend.cu backends/solid_color_cuda/SolidColorCudaBackend.h)

# SDL2
find_package(SDL2 REQUIRED)

include_directories(
        ${SDL2_INCLUDE_DIR})
target_link_libraries(
        ray_tracer
        ${SDL2_LIBRARY})

# libpng
find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})
target_link_libraries(ray_tracer ${PNG_LIBRARY})
