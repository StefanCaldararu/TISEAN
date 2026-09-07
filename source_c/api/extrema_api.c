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

/* Reentrant core of extrema, factored out of source_c/extrema.c so it has
   no dependency on argv parsing or file-scope globals. The math here (the
   parabola fit through each candidate triple of points and the per-event
   interpolation of every component) is unchanged from main()'s scan loop;
   the only difference is that found extrema are appended to a growable
   array instead of being printed immediately. */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../routines/tsa.h"
#include "../../include/extrema.h"

#define EXTREMA_CHUNK 256

static void append_event(ExtremaResult *result, unsigned long *capacity,
			   double *const *series, unsigned long i,
			   double time, double dtime)
{
  unsigned int j;
  double a, b, c;

  if (result->count == *capacity) {
    *capacity += EXTREMA_CHUNK;
    check_alloc(result->point = (double *)realloc(result->point,
	sizeof(double) * (*capacity) * result->dim));
    check_alloc(result->dt = (double *)realloc(result->dt,
	sizeof(double) * (*capacity)));
  }

  for (j = 0; j < result->dim; j++) {
    a = series[j][i - 1];
    b = (series[j][i] - series[j][i - 2]) / 2.0;
    c = (series[j][i] - 2.0 * series[j][i - 1] + series[j][i - 2]) / 2.0;
    result->point[result->count * result->dim + j] = a + b * time + c * sqr(time);
  }
  result->dt[result->count] = dtime;
  result->count++;
}

ExtremaResult *extrema_find(double *const *series, unsigned long length,
			     unsigned int dim, unsigned int which,
			     char maxima, double mintime)
{
  unsigned long i, capacity;
  double x[3], a, b, c, lasttime, nexttime, time;
  ExtremaResult *result;

  if (which >= dim)
    return NULL;

  check_alloc(result = (ExtremaResult *)malloc(sizeof(ExtremaResult)));
  result->count = 0;
  result->dim = dim;
  result->point = NULL;
  result->dt = NULL;

  if (length < 2)
    return result;

  capacity = EXTREMA_CHUNK;
  check_alloc(result->point = (double *)malloc(sizeof(double) * capacity * dim));
  check_alloc(result->dt = (double *)malloc(sizeof(double) * capacity));

  lasttime = 0.0;
  x[0] = series[which][0];
  x[1] = series[which][1];
  for (i = 2; i < length; i++) {
    x[2] = series[which][i];
    if (maxima) {
      if ((x[1] >= x[0]) && (x[1] > x[2])) {
	a = x[1];
	b = (x[2] - x[0]) / 2.0;
	c = (x[2] - 2.0 * x[1] + x[0]) / 2.0;
	time = -b / 2.0 / c;
	nexttime = (double)i - 1.0 + time;
	if ((nexttime - lasttime) >= mintime) {
	  append_event(result, &capacity, series, i, time, nexttime - lasttime);
	  lasttime = nexttime;
	}
      }
    }
    else {
      if ((x[1] <= x[0]) && (x[1] < x[2])) {
	a = x[1];
	b = (x[2] - x[0]) / 2.0;
	c = (x[2] - 2.0 * x[1] + x[0]) / 2.0;
	time = -b / 2.0 / c;
	nexttime = (double)i - 1.0 + time;
	if ((nexttime - lasttime) >= mintime) {
	  append_event(result, &capacity, series, i, time, nexttime - lasttime);
	  lasttime = nexttime;
	}
      }
    }
    x[0] = x[1];
    x[1] = x[2];
  }

  if (result->count == 0) {
    free(result->point);
    free(result->dt);
    result->point = NULL;
    result->dt = NULL;
  }

  return result;
}

void extrema_free(ExtremaResult *result)
{
  if (result == NULL)
    return;
  free(result->point);
  free(result->dt);
  free(result);
}
