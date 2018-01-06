cmake_minimum_required(VERSION 3.9)
project(ray_tracer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${ray_tracer_SOURCE_DIR}/cmake")

# CUDA
find_package(CUDA REQUIRED)

# Project files
set(ray_tracer_SOURCE_DIR ${ray_tracer_SOURCE_DIR}/CUDA-RayTracer)
include_directories(${ray_tracer_SOURCE_DIR})

FILE(GLOB_RECURSE SOURCES
        ${ray_tracer_SOURCE_DIR}/*.cpp
        ${ray_tracer_SOURCE_DIR}/*.cu
        ${ray_tracer_SOURCE_DIR}/*.h
        ${ray_tracer_SOURCE_DIR}/*.hpp)

cuda_add_executable(ray_tracer ${SOURCES})

# SDL2
find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})
link_libraries(${SDL2_LIBDIR})
target_link_libraries(ray_tracer SDL2)

# libpng
find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})
target_link_libraries(ray_tracer ${PNG_LIBRARY})

# Boost
find_package(Boost REQUIRED COMPONENTS program_options filesystem)
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(ray_tracer ${Boost_LIBRARIES})
