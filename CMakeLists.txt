cmake_minimum_required(VERSION 3.9)
project(ray_tracer)

set(CMAKE_CXX_STANDARD 17)

# CUDA
find_package(CUDA REQUIRED)

# Project files
include_directories(${ray_tracer_SOURCE_DIR})

cuda_add_executable(
        ray_tracer
        main.cpp
        frontends/sdl/SDLFrontend.cpp frontends/sdl/SDLFrontend.h
        frontends/sdl/SDLError.cpp frontends/sdl/SDLError.h
        backends/Backend.cpp backends/Backend.h
        backends/solid_color_cuda/SolidColorCudaBackend.cpp backends/solid_color_cuda/SolidColorCudaBackend.cu backends/solid_color_cuda/SolidColorCudaBackend.h)

# SDL2
include(FindPkgConfig)

pkg_search_module(SDL2 REQUIRED sdl2)
pkg_search_module(SDL2IMAGE REQUIRED SDL2_image>=2.0.0)

include_directories(${SDL2_INCLUDE_DIRS} ${SDL2IMAGE_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${SDL2_LIBRARIES} ${SDL2IMAGE_LIBRARIES})
