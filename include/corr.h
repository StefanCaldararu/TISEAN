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

/* Reentrant API for the corr routine: estimates the autocorrelation of a
   data set. Extracted out of source_c/corr.c so it can be called both from
   the corr CLI and from other bindings (e.g. Python) without going through
   global state, argv parsing, or the process-exiting error path in
   variance(). */

#ifndef _CORR_H
#define _CORR_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  unsigned long length;  /* length of the input series actually used */
  unsigned long tau;     /* maximum lag actually computed (clamped to
			     length-1 if the requested tau was too large) */
  double average;        /* mean of the raw (un-centered) series */
  double stddev;         /* standard deviation of the raw series */
  double *values;        /* [tau+1] correlation values for lags 0..tau */
} CorrResult;

/* Computes the autocorrelation of series[0..length-1] for lags 0..tau (tau
   is clamped to length-1 if tau >= length). If normalize is non-zero, the
   series is centered by its own mean before the autocovariance is computed
   and each lag is divided by the variance, matching the CLI's default
   behavior; if zero, the raw (uncentered) series is used and no division is
   applied, matching the CLI's -n flag. series is not modified.

   Returns NULL if length == 0 or the data is constant (variance == 0),
   mirroring variance()'s hard-exit case but without exiting the process. */
CorrResult *corr_compute(const double *series, unsigned long length,
			  unsigned long tau, int normalize);
void corr_free(CorrResult *result);

#ifdef __cplusplus
}
#endif

#endif
