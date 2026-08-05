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

/* Reentrant API for the extrema routine: finds local maxima (or minima) of
   one component of a possibly multivariate series by fitting a parabola
   through each candidate triple of points. Extracted out of
   source_c/extrema.c so it can be called both from the extrema CLI and
   from other bindings (e.g. Python) without going through global state or
   argv parsing. */

#ifndef _EXTREMA_H
#define _EXTREMA_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  unsigned long count;  /* number of extrema found */
  unsigned int dim;      /* number of components per extremum (== input dim) */
  double *point;         /* [count*dim] interpolated series values at each
			     extremum, row-major: point[event*dim+component].
			     NULL if count == 0 */
  double *dt;             /* [count] time since the previous extremum (since
			      t=0 for the first one). NULL if count == 0 */
} ExtremaResult;

/* Scans series[which][0..length-1] (series is [dim][length]) for local
   maxima (maxima != 0) or minima (maxima == 0) by fitting a parabola
   through each candidate triple of consecutive points, the same way the
   extrema CLI does it. Two extrema closer together in interpolated time
   than mintime are merged: only the first of such a pair is kept. For
   every extremum found, every one of the dim components of the series is
   interpolated at the extremum's fractional time via the same parabola
   fit.

   which must be a valid 0-based component index (which < dim); this is
   the reentrant equivalent of the extrema CLI's own "-w has to be
   smaller or equal to the number of components" check. Returns NULL if
   which >= dim.

   If length < 2, there are not even two points to seed the scan from -
   this returns a non-NULL result with count == 0 rather than the
   out-of-bounds array access extrema.c's own unconditional first read
   would suffer from in that case, since a library entry point can be
   handed arbitrary array lengths the CLI's own input files never
   exercised.

   Caller must free the result with extrema_free(). */
ExtremaResult *extrema_find(double *const *series, unsigned long length,
			     unsigned int dim, unsigned int which,
			     char maxima, double mintime);

void extrema_free(ExtremaResult *result);

#ifdef __cplusplus
}
#endif

#endif
