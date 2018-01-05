# raytracer
Whitted ray tracing in CUDA

## Building
### CMake
Setting the compiler version for CUDA explicitly may be needed as CUDA 9
does not support GCC 7. The following CMake option may be added to achieve that:
```
-DCUDA_NVCC_FLAGS="-ccbin /usr/bin/gcc-6"
```
### Visual Studio
Open solution in Visual Studio 2015 or newer, Nvidia CUDA plugin is required.
If prompted to update the toolset settings - decline.
