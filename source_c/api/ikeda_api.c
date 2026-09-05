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

/* Reentrant API for ikeda: iterates the Ikeda map from a given starting
   point. Extracted out of source_c/ikeda.c so it can be called both from
   the ikeda CLI and from other bindings (e.g. Python) without going
   through global state or argv parsing. */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../routines/tsa.h"
#include "../../include/ikeda.h"

double *ikeda_generate(double a, double b, double c, double x0, double y0,
			unsigned long length, unsigned long ntrans)
{
  long n;
  double xo, yo, xn, yn, s, cs, ss, *out;

  if (length == 0)
    return NULL;

  check_alloc(out = (double *)malloc(sizeof(double) * 2 * length));

  xo = x0;
  yo = y0;

  /* n here runs one below the Fortran loop counter (which increments
     before computing): Fortran's n=-ntrans+1..nmax corresponds to
     n=-ntrans..length-1 here, with the write condition n_fortran>=1
     becoming n>=0 and the write index n_fortran-1 becoming n directly. */
  for (n = -(long)ntrans; n < (long)length; n++) {
    s = a - b / (1.0 + xo * xo + yo * yo);
    cs = cos(s);
    ss = sin(s);
    xn = 1.0 + c * (xo * cs - yo * ss);
    yn = c * (xo * ss + yo * cs);
    xo = xn;
    yo = yn;

    if (n < 0)
      continue;
    /* write(iunit,*) real(xn), real(yn) -- the written values are
       rounded to single precision; the state (xo,yo) carried into the
       next iteration is not. */
    out[2 * n] = (double)(float)xn;
    out[2 * n + 1] = (double)(float)yn;
  }

  return out;
}

void ikeda_free(double *series)
{
  free(series);
}
