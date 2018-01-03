# raytracer
Whitted ray tracing in CUDA

## Building
Setting the compiler version for CUDA explicitly may be needed as CUDA 9
does not support GCC 7. The following CMake option may be added to achieve that:
```
-DCUDA_NVCC_FLAGS="-ccbin /usr/bin/gcc-6"
```

When using Windows, set the following environment variables before building
the project:
```
SDL2DIR
SDL2IMAGEDIR
```

## Third-party
* CMake scripts in `cmake` directory: [sdl2-cmake-scripts](https://github.com/tcbrindle/sdl2-cmake-scripts)
