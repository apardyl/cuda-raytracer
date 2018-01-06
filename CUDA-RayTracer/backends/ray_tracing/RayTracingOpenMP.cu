#include <cuda_runtime.h>
#include "RayTracingOpenMP.h"

const int BLOCK_SIZE = 32;
const int BYTES_PER_PIXEL = 3;

