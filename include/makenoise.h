/*
 *   This file is part of TISEAN
 *
 *   Copyright (c) 1998-2007 Rainer Hegger, Holger Kantz, Thomas Schreiber
 *
 *   TISEAN is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation; either version 2 of the License, or
 *   (at your option) any later version.
 *
 *   TISEAN is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with TISEAN; if not, write to the Free Software
 *   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/* Reentrant API for the makenoise routine: adds uniform or Gaussian noise
   to a multi-column time series. Extracted out of source_c/makenoise.c so
   it can be called both from the makenoise CLI and from other bindings
   (e.g. Python) without going through global state, argv parsing, or the
   process-exiting error path in variance(). */

#ifndef _MAKENOISE_H
#define _MAKENOISE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  unsigned int dim;
  unsigned long length;
  double **series;  /* [dim][length], series with noise added */
} MakeNoise;

/* series is [dim][length] and is not modified; the noisy result is a new
   copy. For each column, uniform noise (via rnd_1279()) or, if gaussian is
   non-zero, Gaussian noise (via gaussian()) is added in place on that
   copy, scaled either relative to the column's own standard deviation
   (absolute == 0) or by the fixed noiselevel (absolute != 0) - the same
   way the makenoise CLI's equidistri()/gauss() do it. seed is passed to
   rnd_init() and the generator is warmed up with 10000 discarded draws
   first, exactly like the CLI.
   Returns NULL if dim == 0 or length == 0, or if absolute == 0 and any
   column has zero variance (mirrors variance()'s exit case, without
   exiting the process). */
MakeNoise *makenoise_add(double *const *series, unsigned long length,
			  unsigned int dim, double noiselevel, char absolute,
			  char gaussian, unsigned long seed);
void makenoise_free(MakeNoise *noise);

#ifdef __cplusplus
}
#endif

#endif
