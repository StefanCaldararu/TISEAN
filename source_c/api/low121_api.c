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

/* Reentrant core of low121, factored out of source_c/low121.c so it has no
   dependency on argv parsing or file-scope globals. The math here is
   unchanged from the original inline filter loop. */

#include <stdlib.h>
#include "../routines/tsa.h"
#include "../../include/low121.h"

void low121_pass(const double *series, unsigned long length, double *out)
{
  unsigned long i;

  out[0] = (2.0 * series[0] + 2.0 * series[1]) / 4.0;
  out[length - 1] = (2.0 * series[length - 1] + 2.0 * series[length - 2]) / 4.0;
  for (i = 1; i < length - 1; i++)
    out[i] = (series[i - 1] + 2.0 * series[i] + series[i + 1]) / 4.0;
}

void low121_filter(const double *series, unsigned long length,
		    unsigned int iterations, double *out)
{
  unsigned long i;
  unsigned int iter;
  double *scratch;

  for (i = 0; i < length; i++)
    out[i] = series[i];

  if (iterations == 0)
    return;

  check_alloc(scratch = (double *)malloc(sizeof(double) * length));
  for (iter = 0; iter < iterations; iter++) {
    low121_pass(out, length, scratch);
    for (i = 0; i < length; i++)
      out[i] = scratch[i];
  }
  free(scratch);
}
