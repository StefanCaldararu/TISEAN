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

/* Reentrant core of ikeda, factored out of source_c/ikeda.c so it can be
   called both from the ikeda CLI and from other bindings (e.g. Python)
   without going through global state or argv parsing. The math here is
   unchanged from the original Ikeda map iteration loop. Only the finite
   (explicit -l) case is exposed here: the CLI's -l0 "stream forever" mode
   can't be expressed as a bounded return, so it keeps its own loop in
   ikeda.c. */

#ifndef _IKEDA_H
#define _IKEDA_H

#ifdef __cplusplus
extern "C" {
#endif

/* Iterates the Ikeda map
     s      = a - b/(1 + x_n^2 + y_n^2)
     x_{n+1} = 1 + c*(x_n*cos(s) - y_n*sin(s))
     y_{n+1} =     c*(x_n*sin(s) + y_n*cos(s))
   starting from (x0,y0), discarding the first ntrans transient steps, and
   returns a newly allocated array of 2*length interleaved (x,y) values
   (free with ikeda_free). out[2*n] and out[2*n+1] are the x and y of the
   (n+1)-th point after the transient. Returns NULL if length is 0. */
double *ikeda_generate(double a, double b, double c, double x0, double y0,
			unsigned long length, unsigned long ntrans);
void ikeda_free(double *series);

#ifdef __cplusplus
}
#endif

#endif
