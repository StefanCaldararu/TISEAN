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

/* Reentrant API for the resample routine: resamples a series onto a new,
   evenly spaced time grid via local polynomial interpolation. Extracted
   out of source_c/resample.c so it can be called both from the resample
   CLI and from other bindings (e.g. Python) without going through global
   state, argv parsing, or the process-exiting error path in solvele()
   (called internally, via invert_matrix(), by the original). */

#ifndef _RESAMPLE_H
#define _RESAMPLE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  unsigned long length; /* number of resampled output points */
  double *data;          /* [length] resampled values */
} ResampleResult;

/* Resamples series[0..length-1] onto a new time grid: starting at time
   (order+2)/2 (in units of the original sampling interval) and stepping by
   sampletime, each output point is a degree-`order` polynomial interpolant
   fit through the order+1 original samples centered on that time - the
   same algorithm as the resample CLI's -s/-p options. series is not
   modified.

   Returns NULL if the (order+1)x(order+1) interpolation matrix is
   numerically singular (mirrors solvele()'s hard-exit case but without
   exiting the process); this depends only on order, not on the data. If
   the new grid has no points strictly before the end of series, the
   result is non-NULL with length == 0.

   Like the original inline code, this does not itself validate length
   against order: length must be >= (order+1)/2 (integer division) or the
   internal `length - order/2` wraps around (both are unsigned) and the
   loop reads series out of bounds - confirmed to segfault the original
   CLI for a too-short series. Callers that can't guarantee this (e.g. the
   Python binding) must check it themselves before calling. */
ResampleResult *resample_compute(const double *series, unsigned long length,
				  double sampletime, unsigned int order);
void resample_free(ResampleResult *result);

#ifdef __cplusplus
}
#endif

#endif
