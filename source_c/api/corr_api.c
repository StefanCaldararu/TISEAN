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

/* Reentrant core of corr, factored out of source_c/corr.c so it has no
   dependency on argv parsing, file-scope globals, or the process-exiting
   error path in the generic variance() library routine it used to call.
   The math here (mean/variance via a plain sequential sum, optional
   centering, sum-of-products autocovariance per lag) is unchanged from
   main()/corr(). */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../routines/tsa.h"
#include "../../include/corr.h"

CorrResult *corr_compute(const double *series, unsigned long length,
			  unsigned long tau, int normalize)
{
  unsigned long i;
  long j;
  double av, var, h, norm_div, c;
  double *array;
  CorrResult *result;

  if (length == 0)
    return NULL;

  av = var = 0.0;
  for (i = 0; i < length; i++) {
    h = series[i];
    av += h;
    var += h * h;
  }
  av /= (double)length;
  var = sqrt(fabs(var / (double)length - av * av));
  if (var == 0.0)
    return NULL;

  if (tau >= length)
    tau = length - 1;

  check_alloc(array = (double *)malloc(sizeof(double) * length));
  for (i = 0; i < length; i++)
    array[i] = series[i];
  if (normalize) {
    for (i = 0; i < length; i++)
      array[i] -= av;
  }

  check_alloc(result = (CorrResult *)malloc(sizeof(CorrResult)));
  result->length = length;
  result->tau = tau;
  result->average = av;
  result->stddev = var;
  check_alloc(result->values = (double *)malloc(sizeof(double) * (tau + 1)));

  norm_div = normalize ? (var * var) : 1.0;
  for (i = 0; i <= tau; i++) {
    c = 0.0;
    for (j = 0; j < (long)(length - i); j++)
      c += array[j] * array[j + i];
    c /= (double)(length - i);
    result->values[i] = c / norm_div;
  }

  free(array);
  return result;
}

void corr_free(CorrResult *result)
{
  if (result == NULL)
    return;
  free(result->values);
  free(result);
}
