/*
Copyright 2008, 2009 Travis Desell, Dave Przybylo, Nathan Cole,
Boleslaw Szymanski, Heidi Newberg, Carlos Varela, Malik Magdon-Ismail
and Rensselaer Polytechnic Institute.

This file is part of Milkway@Home.

Milkyway@Home is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Milkyway@Home is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Milkyway@Home.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef _EVALUATION_H_
#define _EVALUATION_H_

#include "parameters.h"
#include "star_points.h"

typedef struct
{
    vector stream_a;
    vector stream_c;
    double stream_sigma;
    double stream_sigma_sq2;
} STREAM_CONSTANTS;

typedef struct
{
    double st_prob_int;    /* for kahan summation */
    double st_prob_int_c;
} ST_PROBS;

/* Replacement for mess of double** used before */
typedef struct
{
    double r_point;
    double r_in_mag;
    double r_in_mag2;
    double qw_r3_N;
} R_POINTS;

typedef struct
{
    double irv;
    double reff_xr_rp3;
} R_CONSTANTS;

typedef struct
{
    double nu;
    double id;
} NU_CONSTANTS;

/* Scratch space used by each integral */
typedef struct
{
    R_CONSTANTS* r_step_consts;
    R_POINTS* rss;
    NU_CONSTANTS* nu_st;
} INTEGRAL_CONSTANTS;

typedef struct
{
    double bg_int;
    double correction;   /* Correction for Kahan summation */
} BG_PROB;

#define CLEAR_BG_PROB(bgp) { (bgp).bg_int = 0.0; (bgp).correction = 0.0; }

/* Add b to a */
#define INCADD_BG_PROB(a, b) { (a).bg_int += (b).bg_int; (a).correction += (b).correction; }

#define ZERO_BG_PROB { 0.0, 0.0 }

typedef struct
{
    double st_only_sum;
    double st_only_sum_c;
} ST_SUM;


typedef struct
{
    double* dx;
    double* qgaus_W;
} STREAM_GAUSS;

double cpu_evaluate(const ASTRONOMY_PARAMETERS* ap,
                    const STAR_POINTS* sp,
                    const STREAMS* streams,
                    const STREAM_CONSTANTS* sc);

STREAM_CONSTANTS* init_constants(ASTRONOMY_PARAMETERS* ap,
                                 const BACKGROUND_PARAMETERS* bgp,
                                 const STREAMS* streams);

void free_constants(ASTRONOMY_PARAMETERS* ap);

#endif /* _EVALUATION_H_ */

