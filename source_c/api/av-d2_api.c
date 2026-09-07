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

/* Reentrant core of av-d2, factored out of source_c/av-d2.c so it has no
   dependency on argv parsing, file-scope globals, or file I/O. The math
   here (the centered moving average over a (2*aver+1)-point window) is
   unchanged from main()'s per-block averaging loop. */

#include <stdio.h>
#include <stdlib.h>
#include "../routines/tsa.h"
#include "../../include/av-d2.h"

AvD2Result *av_d2_average(const double *eps, const double *y,
			   unsigned long howmany, int aver)
{
  long k, j;
  double avy, aveps, norm;
  AvD2Result *result;

  if (eps == NULL || y == NULL || aver < 0)
    return NULL;

  check_alloc(result = (AvD2Result *)malloc(sizeof(AvD2Result)));
  result->avg_eps = NULL;
  result->avg_y = NULL;
  result->n_points = 0;

  if ((unsigned long)(2L * aver + 1L) > howmany)
    return result;

  norm = 2.0 * aver + 1.0;
  result->n_points = howmany - (unsigned long)(2 * aver);
  check_alloc(result->avg_eps = (double *)malloc(sizeof(double) * result->n_points));
  check_alloc(result->avg_y = (double *)malloc(sizeof(double) * result->n_points));

  for (k = aver; k < (long)howmany - aver; k++) {
    avy = aveps = 0.0;
    for (j = -aver; j <= aver; j++) {
      avy += y[k + j];
      aveps += eps[k + j];
    }
    result->avg_eps[k - aver] = aveps / norm;
    result->avg_y[k - aver] = avy / norm;
  }

  return result;
}

void av_d2_free(AvD2Result *result)
{
  if (result == NULL)
    return;
  free(result->avg_eps);
  free(result->avg_y);
  free(result);
}
