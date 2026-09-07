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

/* Reentrant core of xzero, factored out of source_c/xzero.c so it has no
   dependency on argv parsing, file-scope globals, or the process-exiting
   error paths in the generic rescale_data()/variance() library routines it
   used to call. The math here (independent [0,1) rescaling of both series,
   box-assisted delay-embedding neighbor search at a growing radius, the
   zeroth-order forecast fit and its RMS error normalized by series2's
   standard deviation) is unchanged from main()/make_fit(). */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../routines/tsa.h"
#include "../../include/xzero.h"

/* Number of boxes for the neighbor search algorithm; not user-configurable
   in the CLI either (xzero.c has no option for it, just the NMAX macro). */
#define XZERO_NMAX 128

static double make_fit(const double *series1, const double *series2,
			const unsigned long *found, unsigned long act,
			unsigned long number, unsigned long istep)
{
  double casted = 0.0;
  unsigned long i;

  for (i = 0; i < number; i++)
    casted += series1[found[i] + istep];
  casted /= number;

  return (casted - series2[act + istep]) * (casted - series2[act + istep]);
}

XZeroResult *xzero_forecast(const double *series1_in, const double *series2_in,
			     unsigned long length, unsigned int dim,
			     unsigned int delay, unsigned long n_ref,
			     int minn, double eps0, double epsf,
			     unsigned int step, int epsset)
{
  unsigned long i, j, actfound, clength;
  double *series1, *series2;
  double min1, max1, min2, max2, interval1, interval2, interval;
  double h, av2, rms2;
  double epsilon;
  long **box, *list;
  unsigned long *found;
  char *done, alldone;
  XZeroResult *result;

  if (series1_in == NULL || series2_in == NULL || length == 0)
    return NULL;

  check_alloc(series1 = (double *)malloc(sizeof(double) * length));
  check_alloc(series2 = (double *)malloc(sizeof(double) * length));
  for (i = 0; i < length; i++) {
    series1[i] = series1_in[i];
    series2[i] = series2_in[i];
  }

  min1 = max1 = series1[0];
  for (i = 1; i < length; i++) {
    if (series1[i] < min1) min1 = series1[i];
    if (series1[i] > max1) max1 = series1[i];
  }
  interval1 = max1 - min1;
  if (interval1 == 0.0) {
    free(series1);
    free(series2);
    return NULL;
  }
  for (i = 0; i < length; i++)
    series1[i] = (series1[i] - min1) / interval1;

  min2 = max2 = series2[0];
  for (i = 1; i < length; i++) {
    if (series2[i] < min2) min2 = series2[i];
    if (series2[i] > max2) max2 = series2[i];
  }
  interval2 = max2 - min2;
  if (interval2 == 0.0) {
    free(series1);
    free(series2);
    return NULL;
  }
  for (i = 0; i < length; i++)
    series2[i] = (series2[i] - min2) / interval2;

  interval = (interval1 + interval2) / 2.0;

  av2 = rms2 = 0.0;
  for (i = 0; i < length; i++) {
    h = series2[i];
    av2 += h;
    rms2 += h * h;
  }
  av2 /= (double)length;
  rms2 = sqrt(fabs(rms2 / (double)length - av2 * av2));
  if (rms2 == 0.0) {
    free(series1);
    free(series2);
    return NULL;
  }

  check_alloc(list = (long *)malloc(sizeof(long) * length));
  check_alloc(found = (unsigned long *)malloc(sizeof(unsigned long) * length));
  check_alloc(done = (char *)malloc(sizeof(char) * length));
  check_alloc(box = (long **)malloc(sizeof(long *) * XZERO_NMAX));
  for (i = 0; i < XZERO_NMAX; i++)
    check_alloc(box[i] = (long *)malloc(sizeof(long) * XZERO_NMAX));
  for (i = 0; i < length; i++)
    done[i] = 0;

  check_alloc(result = (XZeroResult *)malloc(sizeof(XZeroResult)));
  result->steps = step;
  check_alloc(result->error = (double *)malloc(sizeof(double) * step));
  for (i = 0; i < step; i++)
    result->error[i] = 0.0;

  if (epsset)
    eps0 /= interval;

  epsilon = eps0 / epsf;
  clength = (n_ref <= length) ? n_ref - step : length - step;
  result->clength = clength;

  alldone = 0;
  while (!alldone) {
    alldone = 1;
    epsilon *= epsf;
    make_box(series1, box, list, length - step, XZERO_NMAX, dim, delay, epsilon);
    for (i = (dim - 1) * delay; i < clength; i++)
      if (!done[i]) {
	actfound = find_neighbors(series1, box, list, series2 + i, length,
				   XZERO_NMAX, dim, delay, epsilon, found);
	if (actfound >= (unsigned long)minn) {
	  for (j = 1; j <= step; j++)
	    result->error[j - 1] += make_fit(series1, series2, found, i, actfound, j);
	  done[i] = 1;
	}
	alldone &= done[i];
      }
  }

  for (i = 0; i < step; i++)
    result->error[i] = sqrt(result->error[i] / (double)(clength - (dim - 1) * delay)) / rms2;

  for (i = 0; i < XZERO_NMAX; i++)
    free(box[i]);
  free(box);
  free(list);
  free(found);
  free(done);
  free(series1);
  free(series2);

  return result;
}

void xzero_free(XZeroResult *result)
{
  if (result == NULL)
    return;
  free(result->error);
  free(result);
}
