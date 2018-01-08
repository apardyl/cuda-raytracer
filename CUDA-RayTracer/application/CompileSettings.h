#ifndef RAY_TRACER_COMPILESETTINGS_H
#define RAY_TRACER_COMPILESETTINGS_H

#if __has_include(<gtkmm.h>)
#define GTK_ENABLED true
#else
#define GTK_ENABLED false
#endif

#endif //RAY_TRACER_COMPILESETTINGS_H
