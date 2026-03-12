#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <features.h>

#if defined(__linux__) && defined(__GLIBC_PREREQ) && !__GLIBC_PREREQ(2, 38)
extern "C" int __isoc99_vsscanf(const char* buffer, const char* format,
                                va_list args);
extern "C" int __isoc99_vfscanf(FILE* stream, const char* format,
                                va_list args);
extern "C" long __strtol_internal(const char* nptr, char** endptr, int base,
                                  int group);

extern "C" int __isoc23_sscanf(const char* buffer, const char* format, ...)
{
    va_list args;
    va_start(args, format);
    const int result = __isoc99_vsscanf(buffer, format, args);
    va_end(args);
    return result;
}

extern "C" int __isoc23_fscanf(FILE* stream, const char* format, ...)
{
    va_list args;
    va_start(args, format);
    const int result = __isoc99_vfscanf(stream, format, args);
    va_end(args);
    return result;
}

extern "C" long __isoc23_strtol(const char* nptr, char** endptr, int base)
{
    return __strtol_internal(nptr, endptr, base, 0);
}
#endif
