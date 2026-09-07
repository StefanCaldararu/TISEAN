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

/* Reentrant core of henon, factored out of source_c/henon.c so it can be
   called both from the henon CLI and from other bindings (e.g. Python)
   without going through global state or argv parsing. The math here is
   unchanged from the original Henon map iteration loop. Only the finite
   (explicit -l) case is exposed here: the CLI's -l0 "stream forever" mode
   can't be expressed as a bounded return, so it keeps its own loop in
   henon.c. */

#ifndef _HENON_H
#define _HENON_H

#ifdef __cplusplus
extern "C" {
#endif

/* Iterates the Henon map x_{n+1} = 1 - a*x_n^2 + b*y_n, y_{n+1} = x_n,
   starting from (x0,y0), discarding the first ntrans transient steps, and
   returns a newly allocated array of 2*length interleaved (x,y) values
   (free with henon_free). out[2*n] and out[2*n+1] are the x and y of the
   (n+1)-th point after the transient. Returns NULL if length is 0. */
double *henon_generate(double a, double b, double x0, double y0,
			unsigned long length, unsigned long ntrans);
void henon_free(double *series);

#ifdef __cplusplus
}
#endif

#endif
