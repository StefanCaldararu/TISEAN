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

/* Reentrant API for henon: iterates the Henon map from a given starting
   point. Extracted out of source_c/henon.c so it can be called both from
   the henon CLI and from other bindings (e.g. Python) without going
   through global state or argv parsing. */

#include <stdio.h>
#include <stdlib.h>
#include "../routines/tsa.h"
#include "../../include/henon.h"

double *henon_generate(double a, double b, double x0, double y0,
			unsigned long length, unsigned long ntrans)
{
  long n;
  double xo, yo, xn, yn, *out;

  if (length == 0)
    return NULL;

  check_alloc(out = (double *)malloc(sizeof(double) * 2 * length));

  xo = x0;
  yo = y0;
  n = -(long)ntrans;
  do {
    n++;
    /* xn=1.-a*xo**2+b*yo : xo**2 binds tighter than the multiply in
       Fortran, so this groups as 1 - a*(x*x) + b*y, not 1 - (a*x)*x. */
    xn = 1.0 - a * (xo * xo) + b * yo;
    yn = xo;
    xo = xn;
    yo = yn;
    if (n < 1)
      continue;
    /* write(iunit,*) real(xn), real(yn) -- the written values are
       rounded to single precision; the state (xo,yo) carried into the
       next iteration is not. */
    out[2 * (n - 1)] = (double)(float)xn;
    out[2 * (n - 1) + 1] = (double)(float)yn;
  } while ((unsigned long)n < length);

  return out;
}

void henon_free(double *series)
{
  free(series);
}
