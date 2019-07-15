#define FCS_ENABLE_INTRINSIC
#ifdef FCS_ENABLE_INTRINSIC
#include <immintrin.h>
#include <malloc.h>
#endif

#include <omp.h>

#include "parts/openmp-directc-local-one.c"
#include "parts/openmp-directc-local-two.c"
#include "parts/openmp-directc-local-periodic-intrinsicsv2.c"
