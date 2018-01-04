cmake_minimum_required(VERSION 3.9)
project(ray_tracer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${ray_tracer_SOURCE_DIR}/cmake")

# CUDA
find_package(CUDA REQUIRED)

# Project files
include_directories(${ray_tracer_SOURCE_DIR})

set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
cuda_add_executable(
        ray_tracer
        main.cpp
        backends/Image.h
        frontends/Frontend.cpp frontends/Frontend.h
        frontends/frontend_controller/CountDownLatch.cpp frontends/frontend_controller/CountDownLatch.h
        frontends/frontend_controller/FrontendController.cpp frontends/frontend_controller/FrontendController.h
        frontends/sdl/SDLFrontend.cpp frontends/sdl/SDLFrontend.h
        frontends/sdl/SDLError.cpp frontends/sdl/SDLError.h
        backends/Backend.cpp backends/Backend.h
        backends/solid_color_cuda/SolidColorCudaBackend.cpp backends/solid_color_cuda/SolidColorCudaBackend.cu backends/solid_color_cuda/SolidColorCudaBackend.h)

# SDL2
find_package(SDL2 REQUIRED)
find_package(SDL2_image REQUIRED)

include_directories(
		${SDL2_INCLUDE_DIR}
        ${SDL2_IMAGE_INCLUDE_DIR})
target_link_libraries(
		ray_tracer
		${SDL2_LIBRARY}
		${SDL2_IMAGE_LIBRARIES})
