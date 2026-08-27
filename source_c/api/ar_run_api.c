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

/* Reentrant API for ar-run: iterates an AR model given its coefficients and
   driving variance. Extracted out of source_c/ar-run.c so it can be called
   both from the ar-run CLI and from other bindings (e.g. Python) without
   going through global state or argv parsing. */

#include <stdio.h>
#include <stdlib.h>
#include "../routines/tsa.h"
#include "../../include/ar-run.h"

double *ar_run_generate(unsigned int poles, const double *coeff, double var,
			 unsigned long length, unsigned long ntrans,
			 unsigned long seed)
{
  long n, j, nn, idx;
  double *x, *out, xx;

  if (poles < 1)
    return NULL;

  check_alloc(x = (double *)malloc(sizeof(double) * poles));
  for (j = 0; j < (long)poles; j++)
    x[j] = 0.0;

  rnd_init(seed);

  check_alloc(out = (double *)malloc(sizeof(double) * (length ? length : 1)));

  for (n = -(long)ntrans; n < (long)length; n++) {
    nn = (n + (long)ntrans) % (long)poles;
    xx = gaussian(var);

    for (j = 0; j < (long)poles; j++) {
      idx = (nn - j - 1 + (long)poles) % (long)poles;
      xx += coeff[j] * x[idx];
    }

    x[nn] = xx;

    if (n >= 0)
      out[n] = xx;
  }

  free(x);
  return out;
}

void ar_run_free(double *series)
{
  free(series);
}
