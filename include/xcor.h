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

/* Reentrant API for the xcor routine: estimates the crosscorrelation of two
   data sets. Extracted out of source_c/xcor.c so it can be called both from
   the xcor CLI and from other bindings (e.g. Python) without going through
   global state, argv parsing, or the process-exiting error path in
   variance(). */

#ifndef _XCOR_H
#define _XCOR_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  unsigned long length;  /* length of the input series actually used */
  unsigned long tau;     /* maximum lag actually computed (clamped to
			     length-1 if the requested tau was too large or
			     negative) */
  double average1;       /* mean of the raw (un-centered) first series */
  double stddev1;        /* standard deviation of the raw first series */
  double average2;       /* mean of the raw (un-centered) second series */
  double stddev2;        /* standard deviation of the raw second series */
  double *values;        /* [2*tau+1] crosscorrelation values for lags
			     -tau..tau; values[lag+tau] is the value at
			     lag `lag` */
} XcorResult;

/* Computes the crosscorrelation of series1[0..length-1] against
   series2[0..length-1] for lags -tau..tau (tau is clamped to length-1 if
   tau >= length, which also catches a negative tau since it is compared as
   unsigned). Neither series is modified.

   Returns NULL if length == 0 or either series is constant (variance ==
   0), mirroring variance()'s hard-exit case but without exiting the
   process. */
XcorResult *xcor_compute(const double *series1, const double *series2,
			  unsigned long length, long tau);
void xcor_free(XcorResult *result);

#ifdef __cplusplus
}
#endif

#endif
