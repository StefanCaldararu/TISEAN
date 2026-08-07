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

/* Reentrant core of lyap_k, factored out of source_c/lyap_k.c so it has no
   dependency on argv parsing, file-scope globals, or the process-exiting
   error path in the generic rescale_data() library routine it used to
   call. The math here (the box-assisted nearest-neighbor search in
   put_in_boxes()/lfind_neighbors() and the per-reference-point divergence
   accumulation in iterate_points()) is unchanged from the original.*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../routines/tsa.h"
#include "../../include/lyap_k.h"

#define BOX 128
#define IBOX (BOX-1)

typedef struct {
  const double *series;
  long **box;
  long *liste;
  long *found;
  long **lfound;
  long **count;
  double **lyap;
  unsigned int maxdim, mindim, delay, maxiter, window;
  unsigned long length;
} LyapKState;

static void put_in_boxes(LyapKState *st, double eps)
{
  unsigned long i, blength;
  long j, k;
  const double *series = st->series;
  long **box = st->box;
  long *liste = st->liste;

  blength = st->length - (st->maxdim - 1) * st->delay - st->maxiter;

  for (i = 0; i < BOX; i++)
    for (j = 0; j < BOX; j++)
      box[i][j] = -1;

  for (i = 0; i < blength; i++) {
    j = (long)(series[i] / eps) & IBOX;
    k = (long)(series[i + st->delay] / eps) & IBOX;
    liste[i] = box[j][k];
    box[j][k] = i;
  }
}

static void lfind_neighbors(LyapKState *st, long act, double eps)
{
  unsigned int hi, k, k1;
  long i, j, i1, i2, j1, element;
  long lwindow;
  double dx, eps2 = sqr(eps);
  const double *series = st->series;
  long **box = st->box;
  long *liste = st->liste;
  long *found = st->found;
  long **lfound = st->lfound;
  unsigned int maxdim = st->maxdim, delay = st->delay;

  lwindow = (long)st->window;
  for (hi = 0; hi < maxdim - 1; hi++)
    found[hi] = 0;
  i = (long)(series[act] / eps) & IBOX;
  j = (long)(series[act + delay] / eps) & IBOX;
  for (i1 = i - 1; i1 <= i + 1; i1++) {
    i2 = i1 & IBOX;
    for (j1 = j - 1; j1 <= j + 1; j1++) {
      element = box[i2][j1 & IBOX];
      while (element != -1) {
	if ((element < (act - lwindow)) || (element > (act + lwindow))) {
	  dx = sqr(series[act] - series[element]);
	  if (dx <= eps2) {
	    for (k = 1; k < maxdim; k++) {
	      k1 = k * delay;
	      dx += sqr(series[act + k1] - series[element + k1]);
	      if (dx <= eps2) {
		k1 = k - 1;
		lfound[k1][found[k1]] = element;
		found[k1]++;
	      }
	      else
		break;
	    }
	  }
	}
	element = liste[element];
      }
    }
  }
}

static void iterate_points(LyapKState *st, long act)
{
  double **lfactor;
  double *dx;
  unsigned int i, j, l, l1;
  long k, element, **lcount;
  const double *series = st->series;
  long *found = st->found;
  long **lfound = st->lfound;
  long **count = st->count;
  double **lyap = st->lyap;
  unsigned int maxdim = st->maxdim, mindim = st->mindim, delay = st->delay;
  unsigned int maxiter = st->maxiter;

  check_alloc(lfactor = (double **)malloc(sizeof(double *) * (maxdim - 1)));
  check_alloc(lcount = (long **)malloc(sizeof(long *) * (maxdim - 1)));
  for (i = 0; i < maxdim - 1; i++) {
    check_alloc(lfactor[i] = (double *)malloc(sizeof(double) * (maxiter + 1)));
    check_alloc(lcount[i] = (long *)malloc(sizeof(long) * (maxiter + 1)));
  }
  check_alloc(dx = (double *)malloc(sizeof(double) * (maxiter + 1)));

  for (i = 0; i <= maxiter; i++)
    for (j = 0; j < maxdim - 1; j++) {
      lfactor[j][i] = 0.0;
      lcount[j][i] = 0;
    }

  for (j = mindim - 2; j < maxdim - 1; j++) {
    for (k = 0; k < found[j]; k++) {
      element = lfound[j][k];
      for (i = 0; i <= maxiter; i++)
	dx[i] = sqr(series[act + i] - series[element + i]);
      for (l = 1; l < j + 2; l++) {
	l1 = l * delay;
	for (i = 0; i <= maxiter; i++)
	  dx[i] += sqr(series[act + i + l1] - series[element + l1 + i]);
      }
      for (i = 0; i <= maxiter; i++)
	if (dx[i] > 0.0) {
	  lcount[j][i]++;
	  lfactor[j][i] += dx[i];
	}
    }
  }
  for (i = mindim - 2; i < maxdim - 1; i++)
    for (j = 0; j <= maxiter; j++)
      if (lcount[i][j]) {
	count[i][j]++;
	lyap[i][j] += log(lfactor[i][j] / lcount[i][j]) / 2.0;
      }

  for (i = 0; i < maxdim - 1; i++) {
    free(lfactor[i]);
    free(lcount[i]);
  }
  free(lcount);
  free(lfactor);
  free(dx);
}

LyapK *lyap_k_compute(const double *series_in, unsigned long length,
		       unsigned int mindim, unsigned int maxdim,
		       unsigned int delay,
		       double epsmin, double epsmax,
		       int eps0set, int eps1set,
		       unsigned int epscount,
		       unsigned long reference,
		       unsigned int maxiter,
		       unsigned int window,
		       LyapKProgressFn progress, void *user_data,
		       LyapKError *error)
{
  unsigned long i;
  unsigned int d, j, l, ndim;
  double eps_fak, epsilon, min, max;
  double *series;
  LyapKState st;
  LyapK *result;

  if (error != NULL)
    *error = LYAP_K_OK;

  /* rescale_data(series,length,&min,&max), on a private copy; max here is
     the data's raw interval (max-min), matching the CLI's own variable
     naming. */
  check_alloc(series = (double *)malloc(sizeof(double) * length));
  for (i = 0; i < length; i++)
    series[i] = series_in[i];

  min = max = series[0];
  for (i = 1; i < length; i++) {
    if (series[i] < min) min = series[i];
    if (series[i] > max) max = series[i];
  }
  max -= min;
  if (max == 0.0) {
    free(series);
    if (error != NULL)
      *error = LYAP_K_ERR_ZERO_INTERVAL;
    return NULL;
  }
  for (i = 0; i < length; i++)
    series[i] = (series[i] - min) / max;

  if (eps0set)
    epsmin /= max;
  if (eps1set)
    epsmax /= max;

  if (epsmin >= epsmax) {
    epsmax = epsmin;
    epscount = 1;
  }

  /* Uses the caller's raw (un-clamped) maxdim, exactly like the CLI: the
     too-few-points check below runs before mindim/maxdim get clamped. */
  if (reference > (length - maxiter - (maxdim - 1) * delay))
    reference = length - maxiter - (maxdim - 1) * delay;
  if ((maxiter + (maxdim - 1) * delay) >= length) {
    free(series);
    if (error != NULL)
      *error = LYAP_K_ERR_TOO_FEW_POINTS;
    return NULL;
  }

  if (maxdim < 2)
    maxdim = 2;
  if (mindim < 2)
    mindim = 2;
  if (mindim > maxdim)
    maxdim = mindim;

  ndim = maxdim - mindim + 1;

  check_alloc(st.liste = (long *)malloc(sizeof(long) * length));
  check_alloc(st.found = (long *)malloc(sizeof(long) * (maxdim - 1)));
  check_alloc(st.lfound = (long **)malloc(sizeof(long *) * (maxdim - 1)));
  for (i = 0; i < maxdim - 1; i++)
    check_alloc(st.lfound[i] = (long *)malloc(sizeof(long) * length));
  check_alloc(st.count = (long **)malloc(sizeof(long *) * (maxdim - 1)));
  for (i = 0; i < maxdim - 1; i++)
    check_alloc(st.count[i] = (long *)malloc(sizeof(long) * (maxiter + 1)));
  check_alloc(st.lyap = (double **)malloc(sizeof(double *) * (maxdim - 1)));
  for (i = 0; i < maxdim - 1; i++)
    check_alloc(st.lyap[i] = (double *)malloc(sizeof(double) * (maxiter + 1)));
  check_alloc(st.box = (long **)malloc(sizeof(long *) * BOX));
  for (i = 0; i < BOX; i++)
    check_alloc(st.box[i] = (long *)malloc(sizeof(long) * BOX));

  st.series = series;
  st.maxdim = maxdim;
  st.mindim = mindim;
  st.delay = delay;
  st.maxiter = maxiter;
  st.window = window;
  st.length = length;

  if (epscount == 1)
    eps_fak = 1.0;
  else
    eps_fak = pow(epsmax / epsmin, 1.0 / (double)(epscount - 1));

  check_alloc(result = (LyapK *)malloc(sizeof(LyapK)));
  result->epscount = epscount;
  result->mindim = mindim;
  result->maxdim = maxdim;
  result->maxiter = maxiter;
  check_alloc(result->epsilon = (double *)malloc(sizeof(double) * epscount));
  check_alloc(result->count = (long ***)malloc(sizeof(long **) * epscount));
  check_alloc(result->lyap = (double ***)malloc(sizeof(double **) * epscount));

  for (l = 0; l < epscount; l++) {
    epsilon = epsmin * pow(eps_fak, (double)l);

    for (i = 0; i < maxdim - 1; i++)
      for (j = 0; j <= maxiter; j++) {
	st.count[i][j] = 0;
	st.lyap[i][j] = 0.0;
      }

    put_in_boxes(&st, epsilon);
    for (i = 0; i < reference; i++) {
      lfind_neighbors(&st, i, epsilon);
      iterate_points(&st, i);
    }

    result->epsilon[l] = epsilon * max;
    if (progress != NULL)
      progress(result->epsilon[l], user_data);

    check_alloc(result->count[l] = (long **)malloc(sizeof(long *) * ndim));
    check_alloc(result->lyap[l] = (double **)malloc(sizeof(double *) * ndim));
    for (d = 0; d < ndim; d++) {
      check_alloc(result->count[l][d] = (long *)malloc(sizeof(long) * (maxiter + 1)));
      check_alloc(result->lyap[l][d] = (double *)malloc(sizeof(double) * (maxiter + 1)));
      for (j = 0; j <= maxiter; j++) {
	result->count[l][d][j] = st.count[mindim - 2 + d][j];
	result->lyap[l][d][j] = st.lyap[mindim - 2 + d][j];
      }
    }
  }

  for (i = 0; i < maxdim - 1; i++) {
    free(st.lfound[i]);
    free(st.count[i]);
    free(st.lyap[i]);
  }
  free(st.lfound);
  free(st.count);
  free(st.lyap);
  free(st.liste);
  free(st.found);
  for (i = 0; i < BOX; i++)
    free(st.box[i]);
  free(st.box);
  free(series);

  return result;
}

void lyap_k_free(LyapK *result)
{
  unsigned int l, d;

  if (result == NULL)
    return;
  if (result->count != NULL) {
    for (l = 0; l < result->epscount; l++) {
      if (result->count[l] != NULL) {
	for (d = 0; d < result->maxdim - result->mindim + 1; d++)
	  free(result->count[l][d]);
	free(result->count[l]);
      }
    }
    free(result->count);
  }
  if (result->lyap != NULL) {
    for (l = 0; l < result->epscount; l++) {
      if (result->lyap[l] != NULL) {
	for (d = 0; d < result->maxdim - result->mindim + 1; d++)
	  free(result->lyap[l][d]);
	free(result->lyap[l]);
      }
    }
    free(result->lyap);
  }
  free(result->epsilon);
  free(result);
}
