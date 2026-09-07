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

/* Reentrant core of poincare, factored out of source_c/poincare.c so it has
   no dependency on argv parsing, file-scope globals, or the process-exiting
   error path in the generic variance() library routine it used to call.
   The math here (the mean/variance via a plain sequential sum, the min/max
   scan, and the two crossing-detection loops with their linear
   interpolation) is unchanged from main()/poincare(); the only difference
   is that crossings are appended to a growable array instead of being
   printed immediately. */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../routines/tsa.h"
#include "../../include/poincare.h"

#define POINCARE_CHUNK 256

static void append_crossing(PoincareResult *result, unsigned long *capacity,
			     const double *series, long i, int dim, int comp,
			     int delay, double delta, double dtime)
{
  long j, jd;
  unsigned int ncoord = (dim > 1) ? (unsigned int)(dim - 1) : 0, k;

  if (result->count == *capacity) {
    *capacity += POINCARE_CHUNK;
    if (ncoord > 0)
      check_alloc(result->point = (double *)realloc(result->point,
	  sizeof(double) * (*capacity) * ncoord));
    check_alloc(result->dt = (double *)realloc(result->dt,
	sizeof(double) * (*capacity)));
  }

  k = 0;
  for (j = -(comp - 1); j <= dim - comp; j++) {
    if (j != 0) {
      jd = i + j * delay;
      result->point[result->count * ncoord + k] =
	  series[jd] + delta * (series[jd + 1] - series[jd]);
      k++;
    }
  }
  result->dt[result->count] = dtime;
  result->count++;
}

PoincareResult *poincare_compute(const double *series, unsigned long length,
				  int dim, int comp, int delay, int dir,
				  int whereset, double where,
				  double *out_min, double *out_max,
				  PoincareError *error)
{
  unsigned long i, capacity;
  double delta, time, lasttime;
  double average, var, h, min, max;
  PoincareResult *result;

  if (error != NULL)
    *error = POINCARE_OK;

  if (length == 0) {
    if (error != NULL)
      *error = POINCARE_ERR_EMPTY_SERIES;
    return NULL;
  }

  average = var = 0.0;
  for (i = 0; i < length; i++) {
    h = series[i];
    average += h;
    var += h * h;
  }
  average /= (double)length;
  var = sqrt(fabs(var / (double)length - average * average));
  if (var == 0.0) {
    if (error != NULL)
      *error = POINCARE_ERR_ZERO_VARIANCE;
    return NULL;
  }

  min = max = series[0];
  for (i = 1; i < length; i++) {
    if (series[i] < min) min = series[i];
    if (series[i] > max) max = series[i];
  }
  if (out_min != NULL) *out_min = min;
  if (out_max != NULL) *out_max = max;

  if (!whereset)
    where = average;

  if (comp > dim) {
    if (error != NULL)
      *error = POINCARE_ERR_WRONG_COMPONENT;
    return NULL;
  }
  if ((where < min) || (where > max)) {
    if (error != NULL)
      *error = POINCARE_ERR_OUTSIDE_REGION;
    return NULL;
  }

  check_alloc(result = (PoincareResult *)malloc(sizeof(PoincareResult)));
  result->count = 0;
  result->dim = (unsigned int)dim;
  result->point = NULL;
  result->dt = NULL;
  capacity = 0;
  time = 0.0;
  lasttime = 0.0;

  if (dir == 0) {
    for (i = (comp - 1) * delay; i < length - (dim - comp) * delay - 1; i++) {
      if ((series[i] < where) && (series[i + 1] >= where)) {
	delta = (series[i] - where) / (series[i] - series[i + 1]);
	time = (double)i + delta;
	if (lasttime > 0.0)
	  append_crossing(result, &capacity, series, (long)i, dim, comp,
			   delay, delta, time - lasttime);
	lasttime = time;
      }
    }
  }
  else {
    for (i = (comp - 1) * delay; i < length - (dim - comp) * delay - 1; i++) {
      if ((series[i] > where) && (series[i + 1] <= where)) {
	delta = (series[i] - where) / (series[i] - series[i + 1]);
	time = (double)i + delta;
	if (lasttime > 0.0)
	  append_crossing(result, &capacity, series, (long)i, dim, comp,
			   delay, delta, time - lasttime);
	lasttime = time;
      }
    }
  }

  return result;
}

void poincare_free(PoincareResult *result)
{
  if (result == NULL)
    return;
  free(result->point);
  free(result->dt);
  free(result);
}
