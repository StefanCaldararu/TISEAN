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

/* Reentrant core of rescale, factored out of source_c/rescale.c so it has no
   dependency on argv parsing, file-scope globals, or the process-exiting
   error paths in the generic variance()/rescale_data() library routines it
   used to call. The math (mean/variance via a plain sequential sum, and the
   min/max scan and rescale) mirrors those two routines and main()'s per-row
   loop exactly. */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../routines/tsa.h"
#include "../../include/rescale.h"

RescaleResult *rescale_compute(double *const *series, unsigned long length,
				unsigned int dim, int set_av, int set_var,
				double xmin, double xmax, RescaleError *error)
{
  unsigned int n;
  unsigned long i;
  double av, varianz, min, interval, h;
  RescaleResult *result;

  if (error != NULL)
    *error = RESCALE_OK;

  if (length == 0) {
    if (error != NULL)
      *error = RESCALE_ERR_EMPTY_SERIES;
    return NULL;
  }
  if (xmin >= xmax) {
    if (error != NULL)
      *error = RESCALE_ERR_WRONG_INTERVAL;
    return NULL;
  }

  check_alloc(result = (RescaleResult *)malloc(sizeof(RescaleResult)));
  result->dim = dim;
  result->length = length;
  check_alloc(result->data = (double **)malloc(sizeof(double *) * dim));
  for (n = 0; n < dim; n++)
    result->data[n] = NULL;

  for (n = 0; n < dim; n++) {
    av = varianz = 0.0;
    for (i = 0; i < length; i++) {
      h = series[n][i];
      av += h;
      varianz += h * h;
    }
    av /= (double)length;
    varianz = sqrt(fabs(varianz / (double)length - av * av));
    if (varianz == 0.0) {
      if (error != NULL)
	*error = RESCALE_ERR_ZERO_VARIANCE;
      rescale_free(result);
      return NULL;
    }

    check_alloc(result->data[n] = (double *)malloc(sizeof(double) * length));
    for (i = 0; i < length; i++)
      result->data[n][i] = series[n][i];

    if (set_av)
      for (i = 0; i < length; i++)
	result->data[n][i] -= av;

    if (set_var)
      for (i = 0; i < length; i++)
	result->data[n][i] /= varianz;

    if (!set_var && !set_av) {
      min = interval = result->data[n][0];
      for (i = 1; i < length; i++) {
	if (result->data[n][i] < min) min = result->data[n][i];
	if (result->data[n][i] > interval) interval = result->data[n][i];
      }
      interval -= min;
      for (i = 0; i < length; i++)
	result->data[n][i] = (result->data[n][i] - min) / interval * (xmax - xmin) + xmin;
    }
  }

  return result;
}

void rescale_free(RescaleResult *result)
{
  unsigned int n;

  if (result == NULL)
    return;
  if (result->data != NULL) {
    for (n = 0; n < result->dim; n++)
      free(result->data[n]);
    free(result->data);
  }
  free(result);
}
