/* ************************************************************************** */
/* REAL.H: include file to support compile-time specification of precision */
/* in floating-point calculations.  If the DOUBLEPREC symbol is defined to */
/* the preprocessor, calculations are done in double precision; otherwise, */
/* they may be done in single precision. */
/* */
/* Rationale: ANSI C enables programmers to write single-precision code, */
/* but does not make it easy to change the precision of code at compile */
/* time, since different functions names are used for floating and double */
/* calculations.  This package introduces the keyword "real", which may be */
/* either float or double, and defines functions which compute with */
/* real-valued numbers. */
/* */
/* Copyright (c) 1993 by Joshua E. Barnes, Honolulu, HI. */
/* It's free because it's yours. */
/* ************************************************************************** */

#if !defined(_MILKYWAY_MATH_H_INSIDE_) && !defined(MILKYWAY_MATH_COMPILATION)
  #error "Only milkyway_math.h can be included directly."
#endif

#ifndef _REAL_H_
#define _REAL_H_

#include "milkyway_cl.h"

#if (!SEPARATION_OPENCL && !NBODY_OPENCL) || __OPENCL_VERSION__
  /* No opencl, or in the kernel */
  #if DOUBLEPREC
    typedef double real;
  #else
    typedef float real;
  #endif /* DOUBLEPREC */
#else
  /* Should use correctly aligned cl_types for the host */
  #if DOUBLEPREC
    typedef cl_double real;
  #else
    typedef cl_float real;
  #endif /* DOUBLEPREC */
#endif

#ifndef __OPENCL_VERSION__
  #include <tgmath.h>

  #if ENABLE_CRLIBM
    #include <crlibm.h>
  #endif /* ENABLE_CRLIBM */
#endif /* __OPENCL_VERSION__ */

#endif /* _REAL_H_ */
