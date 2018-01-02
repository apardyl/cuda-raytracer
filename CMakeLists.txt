cmake_minimum_required(VERSION 3.9)
project(ray_tracer)

set(CMAKE_CXX_STANDARD 17)

add_executable(
        ray_tracer
        main.cpp
        frontends/sdl/SDLFrontend.cpp frontends/sdl/SDLFrontend.h
        frontends/sdl/SDLError.cpp frontends/sdl/SDLError.h
)

include(FindPkgConfig)

pkg_search_module(SDL2 REQUIRED sdl2)
pkg_search_module(SDL2IMAGE REQUIRED SDL2_image>=2.0.0)

include_directories(${SDL2_INCLUDE_DIRS} ${SDL2IMAGE_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${SDL2_LIBRARIES} ${SDL2IMAGE_LIBRARIES})
