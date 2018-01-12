cmake_minimum_required(VERSION 3.9)
project(ray_tracer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${ray_tracer_SOURCE_DIR}/cmake")

# CUDA
find_package(CUDA)

# Project files
set(ray_tracer_SOURCE_DIR ${ray_tracer_SOURCE_DIR}/CUDA-RayTracer)
include_directories(${ray_tracer_SOURCE_DIR})

FILE(GLOB_RECURSE SOURCES
        ${ray_tracer_SOURCE_DIR}/*.cpp
        ${ray_tracer_SOURCE_DIR}/*.cu
        ${ray_tracer_SOURCE_DIR}/*.h
        ${ray_tracer_SOURCE_DIR}/*.hpp)

if (CUDA_FOUND)
    cuda_add_executable(ray_tracer ${SOURCES})
else (CUDA_FOUND)
    add_executable(ray_tracer ${SOURCES})
endif (CUDA_FOUND)

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif (OPENMP_FOUND)

# Threads
find_package(Threads)
target_link_libraries(ray_tracer ${CMAKE_THREAD_LIBS_INIT})

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

# GTK+
find_package(PkgConfig)
pkg_check_modules(GTKMM gtkmm-3.0)
if (GTKMM_FOUND)
    include_directories(${GTKMM_INCLUDE_DIRS})
    link_directories(${GTKMM_LIBRARY_DIRS})
    target_link_libraries(ray_tracer ${GTKMM_LIBRARIES})
    file(COPY CUDA-RayTracer/frontends/gtk/MainWindow.glade
            DESTINATION ${CMAKE_BINARY_DIR})
endif (GTKMM_FOUND)
