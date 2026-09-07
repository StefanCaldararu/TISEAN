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

/* Reentrant API for the low121 routine: a simple [1,2,1]/4 lowpass filter
   in the time domain. Extracted out of source_c/low121.c so it can be
   called both from the low121 CLI and from other bindings (e.g. Python)
   without going through global state or argv parsing. */

#ifndef _LOW121_H
#define _LOW121_H

#ifdef __cplusplus
extern "C" {
#endif

/* Applies a single pass of the [1,2,1]/4 filter to series[0..length-1],
   writing the result to out[0..length-1]. out must not alias series.
   length must be >= 2. */
void low121_pass(const double *series, unsigned long length, double *out);

/* Applies low121_pass `iterations` times in sequence to series[0..length-1]
   and writes the final result to out[0..length-1] (out may alias series).
   length must be >= 2. iterations == 0 just copies series into out. */
void low121_filter(const double *series, unsigned long length,
		    unsigned int iterations, double *out);

#ifdef __cplusplus
}
#endif

#endif
