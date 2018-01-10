#ifndef RAY_TRACER_COMPILESETTINGS_H
#define RAY_TRACER_COMPILESETTINGS_H

#if defined(__has_include)

#if __has_include(<gtkmm.h>)
#define GTK_ENABLED true
#else
#define GTK_ENABLED false
#endif

#if __has_include(<cuda_runtime.h>)
#define CUDA_ENABLED true
#else
#define CUDA_ENABLED false
#endif

#define CUDA_ENABLED true

#else

// Set manualy if __has_include is not supported
#define GTK_ENABLED false
#define CUDA_ENABLED true

#endif
#endif //RAY_TRACER_COMPILESETTINGS_H
