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

/* Reentrant core of xcor, factored out of source_c/xcor.c so it has no
   dependency on argv parsing, file-scope globals, or the process-exiting
   error path in the generic variance() library routine it used to call.
   The math here (mean/variance via a plain sequential sum, centering,
   sum-of-products crosscovariance per lag) is unchanged from main()/
   corr(). */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../routines/tsa.h"
#include "../../include/xcor.h"

XcorResult *xcor_compute(const double *series1, const double *series2,
			  unsigned long length, long tau)
{
  unsigned long i, utau;
  long j, hi, lag, ltau;
  double av1, var1, av2, var2, h, c;
  double *array1, *array2;
  XcorResult *result;

  if (length == 0)
    return NULL;

  av1 = var1 = 0.0;
  for (i = 0; i < length; i++) {
    h = series1[i];
    av1 += h;
    var1 += h * h;
  }
  av1 /= (double)length;
  var1 = sqrt(fabs(var1 / (double)length - av1 * av1));
  if (var1 == 0.0)
    return NULL;

  av2 = var2 = 0.0;
  for (i = 0; i < length; i++) {
    h = series2[i];
    av2 += h;
    var2 += h * h;
  }
  av2 /= (double)length;
  var2 = sqrt(fabs(var2 / (double)length - av2 * av2));
  if (var2 == 0.0)
    return NULL;

  if ((unsigned long)tau >= length)
    tau = (long)length - 1;
  ltau = tau;
  utau = (unsigned long)ltau;

  check_alloc(array1 = (double *)malloc(sizeof(double) * length));
  check_alloc(array2 = (double *)malloc(sizeof(double) * length));
  for (i = 0; i < length; i++) {
    array1[i] = series1[i] - av1;
    array2[i] = series2[i] - av2;
  }

  check_alloc(result = (XcorResult *)malloc(sizeof(XcorResult)));
  result->length = length;
  result->tau = utau;
  result->average1 = av1;
  result->stddev1 = var1;
  result->average2 = av2;
  result->stddev2 = var2;
  check_alloc(result->values = (double *)malloc(sizeof(double) * (2 * utau + 1)));

  for (lag = -ltau; lag <= ltau; lag++) {
    unsigned long count = 0;

    c = 0.0;
    for (j = 0; j < (long)length; j++) {
      hi = j + lag;
      if (hi >= 0 && hi < (long)length) {
	count++;
	c += array1[j] * array2[hi];
      }
    }
    c /= (double)count;
    result->values[lag + ltau] = c / var1 / var2;
  }

  free(array1);
  free(array2);
  return result;
}

void xcor_free(XcorResult *result)
{
  if (result == NULL)
    return;
  free(result->values);
  free(result);
}
