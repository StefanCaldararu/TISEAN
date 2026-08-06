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

/* Reentrant core of lzo-gm, factored out of source_c/lzo-gm.c so it has no
   dependency on argv parsing, file-scope globals, or the process-exiting
   error path in the generic rescale_data() library routine it used to
   call. The math here (per-dimension rescale to [0,1), the box-assisted
   neighbor search, the local-constant fit and its error accumulation) is
   unchanged from main()/make_fit(); the only difference is that qualifying
   rows (pfound > 1) are appended to a growable array instead of being
   printed immediately. */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../routines/tsa.h"
#include "../../include/lzo-gm.h"

/*number of boxes for the neighbor search algorithm*/
#define NMAX 256
#define LZO_GM_CHUNK 64

static void make_fit(double *const *series, const unsigned long *found,
		      unsigned long number, long act, int step,
		      unsigned int dim, double *error)
{
  double *si, cast;
  unsigned int i;
  unsigned long j;

  for (i = 0; i < dim; i++) {
    si = series[i];
    cast = si[found[0] + step];
    for (j = 1; j < number; j++)
      cast += si[found[j] + step];
    cast /= (double)number;
    error[i] += sqr(cast - series[i][act + step]);
  }
}

LzoGmResult *lzo_gm_compute(double *const *series, unsigned long length,
			     unsigned int dim, unsigned int embed,
			     unsigned int delay, int step,
			     unsigned long causal, unsigned long iterations,
			     double eps0, char eps0_raw,
			     double eps1, char eps1_raw, double epsf,
			     double *bad_value)
{
  unsigned int d;
  unsigned long ui, clength, capacity, n_rows;
  long i, j, pfound;
  unsigned long *hfound, *found_idx;
  long *list, **box;
  double *hrms, *hav, *error;
  double **rescaled, **hser;
  double min, interval, maxinterval, epsilon, avfound, sumerror;
  unsigned long actfound;
  LzoGmResult *result;

  if (dim == 0 || embed == 0 || length == 0 || step < 0 ||
      (unsigned long)step >= length || iterations < (unsigned long)step)
    return NULL;

  check_alloc(rescaled = (double **)malloc(sizeof(double *) * dim));
  maxinterval = 0.0;
  for (d = 0; d < dim; d++) {
    check_alloc(rescaled[d] = (double *)malloc(sizeof(double) * length));
    min = interval = series[d][0];
    for (ui = 1; ui < length; ui++) {
      if (series[d][ui] < min)
	min = series[d][ui];
      if (series[d][ui] > interval)
	interval = series[d][ui];
    }
    interval -= min;
    if (interval == 0.0) {
      if (bad_value != NULL)
	*bad_value = min;
      for (ui = 0; ui <= d; ui++)
	free(rescaled[ui]);
      free(rescaled);
      return NULL;
    }
    for (ui = 0; ui < length; ui++)
      rescaled[d][ui] = (series[d][ui] - min) / interval;
    if (interval > maxinterval)
      maxinterval = interval;
  }
  interval = maxinterval;

  if (eps0_raw)
    eps0 /= interval;
  if (eps1_raw)
    eps1 /= interval;

  clength = (iterations <= length) ? iterations - step : length - step;

  check_alloc(list = (long *)malloc(sizeof(long) * length));
  check_alloc(found_idx = (unsigned long *)malloc(sizeof(unsigned long) * length));
  check_alloc(hfound = (unsigned long *)malloc(sizeof(unsigned long) * length));
  check_alloc(box = (long **)malloc(sizeof(long *) * NMAX));
  for (i = 0; i < NMAX; i++)
    check_alloc(box[i] = (long *)malloc(sizeof(long) * NMAX));
  check_alloc(error = (double *)malloc(sizeof(double) * dim));
  check_alloc(hrms = (double *)malloc(sizeof(double) * dim));
  check_alloc(hav = (double *)malloc(sizeof(double) * dim));
  check_alloc(hser = (double **)malloc(sizeof(double *) * dim));

  capacity = LZO_GM_CHUNK;
  n_rows = 0;
  check_alloc(result = (LzoGmResult *)malloc(sizeof(LzoGmResult)));
  check_alloc(result->epsilon = (double *)malloc(sizeof(double) * capacity));
  check_alloc(result->avg_error = (double *)malloc(sizeof(double) * capacity));
  check_alloc(result->error = (double *)malloc(sizeof(double) * capacity * dim));
  check_alloc(result->fraction = (double *)malloc(sizeof(double) * capacity));
  check_alloc(result->avneighbors = (double *)malloc(sizeof(double) * capacity));

  for (epsilon = eps0; epsilon < eps1 * epsf; epsilon *= epsf) {
    pfound = 0;
    for (d = 0; d < dim; d++)
      error[d] = hrms[d] = hav[d] = 0.0;
    avfound = 0.0;
    make_multi_box(rescaled, box, list, length - step, NMAX, dim,
		    embed, delay, epsilon);
    for (i = (embed - 1) * delay; i < (long)clength; i++) {
      for (j = 0; j < dim; j++)
	hser[j] = rescaled[j] + i;
      actfound = find_multi_neighbors(rescaled, box, list, hser, length,
				       NMAX, dim, embed, delay, epsilon, hfound);
      actfound = exclude_interval(actfound, i - (long)causal + 1,
				   i + (long)causal + (long)((embed - 1) * delay) - 1,
				   hfound, found_idx);
      if (actfound > 2 * (dim * embed + 1)) {
	make_fit(rescaled, found_idx, actfound, i, step, dim, error);
	pfound++;
	avfound += (double)(actfound - 1);
	for (j = 0; j < dim; j++) {
	  hrms[j] += rescaled[j][i + step] * rescaled[j][i + step];
	  hav[j] += rescaled[j][i + step];
	}
      }
    }
    if (pfound > 1) {
      sumerror = 0.0;
      for (j = 0; j < dim; j++) {
	hav[j] /= pfound;
	hrms[j] = sqrt(fabs(hrms[j] / (pfound - 1) - hav[j] * hav[j] * pfound / (pfound - 1)));
	error[j] = sqrt(error[j] / pfound) / hrms[j];
	sumerror += error[j];
      }

      if (n_rows == capacity) {
	capacity += LZO_GM_CHUNK;
	check_alloc(result->epsilon = (double *)realloc(result->epsilon, sizeof(double) * capacity));
	check_alloc(result->avg_error = (double *)realloc(result->avg_error, sizeof(double) * capacity));
	check_alloc(result->error = (double *)realloc(result->error, sizeof(double) * capacity * dim));
	check_alloc(result->fraction = (double *)realloc(result->fraction, sizeof(double) * capacity));
	check_alloc(result->avneighbors = (double *)realloc(result->avneighbors, sizeof(double) * capacity));
      }
      result->epsilon[n_rows] = epsilon * interval;
      result->avg_error[n_rows] = sumerror / (double)dim;
      for (j = 0; j < dim; j++)
	result->error[n_rows * dim + j] = error[j];
      result->fraction[n_rows] = (double)pfound / (double)(clength - (embed - 1) * delay);
      result->avneighbors[n_rows] = avfound / pfound;
      n_rows++;
    }
  }

  free(list);
  free(found_idx);
  free(hfound);
  free(error);
  free(hrms);
  free(hav);
  free(hser);
  for (i = 0; i < NMAX; i++)
    free(box[i]);
  free(box);
  for (d = 0; d < dim; d++)
    free(rescaled[d]);
  free(rescaled);

  result->dim = dim;
  result->n_rows = n_rows;
  return result;
}

void lzo_gm_free(LzoGmResult *result)
{
  if (result == NULL)
    return;
  free(result->epsilon);
  free(result->avg_error);
  free(result->error);
  free(result->fraction);
  free(result->avneighbors);
  free(result);
}
