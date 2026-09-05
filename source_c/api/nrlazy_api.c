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

/* Reentrant core of nrlazy, factored out of source_c/nrlazy.c so it has no
   dependency on argv parsing, file-scope globals, or the process-exiting
   error path in the generic rescale_data() library routine it used to
   call. The math here (per-component rescale to [0,1), the box-assisted
   neighbor search and averaging correction, iterated `iterations` times)
   is unchanged from main()/correct().  */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../routines/tsa.h"
#include "../../include/nrlazy.h"

#define NRLAZY_BOX 512u

/* Reentrant version of the original main()'s correct(): finds the
   neighbors of the embedded point at index n (across all `comp` rescaled
   components) within Chebyshev distance eps, accumulates their average
   into corr/nf, and returns the number of neighbors found. */
static unsigned int correct_point(unsigned long n, double eps,
				   double *const *series, unsigned int comp,
				   unsigned int embed, unsigned int delay,
				   unsigned int alldim,
				   unsigned int *const *indexes,
				   long *const *box, const long *list,
				   double **corr, long **nf, double *hcor)
{
  int i, i1, i2, j, j1;
  unsigned int k, hdel, hcomp;
  int ibox = (int)NRLAZY_BOX - 1;
  double epsinv, dx;
  long element, nfound = 0;

  epsinv = 1. / eps;

  for (k = 0; k < alldim; k++)
    hcor[k] = 0.0;

  i = (int)(series[0][n] * epsinv) & ibox;
  j = (int)(series[comp - 1][n - (embed - 1) * delay] * epsinv) & ibox;

  for (i1 = i - 1; i1 <= i + 1; i1++) {
    i2 = i1 & ibox;
    for (j1 = j - 1; j1 <= j + 1; j1++) {
      element = box[i2][j1 & ibox];
      while (element != -1) {
	for (k = 0; k < alldim; k++) {
	  hcomp = indexes[0][k];
	  hdel = indexes[1][k];
	  dx = fabs(series[hcomp][n - hdel] - series[hcomp][element - hdel]);
	  if (dx > eps)
	    break;
	}
	if (k == alldim) {
	  nfound++;
	  for (k = 0; k < alldim; k++) {
	    hcomp = indexes[0][k];
	    hdel = indexes[1][k];
	    hcor[k] += series[hcomp][element - hdel];
	  }
	}
	element = list[element];
      }
    }
  }
  for (k = 0; k < alldim; k++) {
    hcomp = indexes[0][k];
    hdel = indexes[1][k];
    corr[hcomp][n - hdel] += hcor[k] / nfound;
    nf[hcomp][n - hdel]++;
  }

  return nfound;
}

NRLazyResult *nrlazy_correct(double *const *series, unsigned long length,
			      unsigned int comp, unsigned int embed,
			      unsigned int delay, unsigned int iterations,
			      double eps_r, double eps_v, double *bad_value,
			      NRLazyIterationFn on_iteration,
			      void *on_iteration_data)
{
  unsigned int i, alldim;
  unsigned long n, ui;
  double *cmin, *cinterval, maxinterval, maxdvar, eps;
  double **resc;
  unsigned int **indexes;
  long **box, *list, **nf;
  double **corr, *hcor;
  unsigned int *nmf;
  unsigned int iter;
  NRLazyResult *result;

  check_alloc(resc = (double **)malloc(sizeof(double *) * comp));
  check_alloc(cmin = (double *)malloc(sizeof(double) * comp));
  check_alloc(cinterval = (double *)malloc(sizeof(double) * comp));

  maxinterval = 0.0;
  maxdvar = 0.0;
  for (i = 0; i < comp; i++) {
    double lmin, linterval, lav, lvar, h;

    check_alloc(resc[i] = (double *)malloc(sizeof(double) * length));

    lmin = linterval = series[i][0];
    for (n = 1; n < length; n++) {
      if (series[i][n] < lmin) lmin = series[i][n];
      if (series[i][n] > linterval) linterval = series[i][n];
    }
    linterval -= lmin;

    if (linterval == 0.0) {
      if (bad_value != NULL)
	*bad_value = lmin;
      for (ui = 0; ui <= i; ui++)
	free(resc[ui]);
      free(resc);
      free(cmin);
      free(cinterval);
      return NULL;
    }

    for (n = 0; n < length; n++)
      resc[i][n] = (series[i][n] - lmin) / linterval;
    cmin[i] = lmin;
    cinterval[i] = linterval;
    if (linterval > maxinterval)
      maxinterval = linterval;

    /* variance() computed on the now-rescaled component, matching the
       original ordering (rescale_data() then variance() per component). */
    lav = lvar = 0.0;
    for (n = 0; n < length; n++) {
      h = resc[i][n];
      lav += h;
      lvar += h * h;
    }
    lav /= (double)length;
    lvar = sqrt(fabs(lvar / (double)length - lav * lav));
    if (lvar > maxdvar)
      maxdvar = lvar;
  }

  if (!isnan(eps_v))
    eps = eps_v * maxdvar;
  else if (!isnan(eps_r))
    eps = eps_r / maxinterval;
  else
    eps = 1.0 / 1000.;

  alldim = comp * embed;
  indexes = make_multi_index(comp, embed, delay);

  check_alloc(nmf = (unsigned int *)malloc(sizeof(unsigned int) * length));
  check_alloc(list = (long *)malloc(sizeof(long) * length));
  check_alloc(box = (long **)malloc(sizeof(long *) * NRLAZY_BOX));
  for (n = 0; n < NRLAZY_BOX; n++)
    check_alloc(box[n] = (long *)malloc(sizeof(long) * NRLAZY_BOX));

  check_alloc(nf = (long **)malloc(sizeof(long *) * comp));
  check_alloc(corr = (double **)malloc(sizeof(double *) * comp));
  for (i = 0; i < comp; i++) {
    check_alloc(nf[i] = (long *)malloc(sizeof(long) * length));
    check_alloc(corr[i] = (double *)malloc(sizeof(double) * length));
  }
  check_alloc(hcor = (double *)malloc(sizeof(double) * alldim));

  for (iter = 1; iter <= iterations; iter++) {
    make_multi_box2(resc, box, list, length, NRLAZY_BOX, comp, embed, delay, eps);
    for (n = 0; n < length; n++) {
      for (i = 0; i < comp; i++) {
	corr[i][n] = 0.0;
	nf[i][n] = 0;
      }
      nmf[n] = 1;
    }

    for (n = (unsigned long)(embed - 1) * delay; n < length; n++)
      nmf[n] = correct_point(n, eps, (double *const *)resc, comp, embed,
			      delay, alldim, (unsigned int *const *)indexes,
			      (long *const *)box, list, corr, nf, hcor);

    for (n = 0; n < length; n++)
      for (i = 0; i < comp; i++)
	if (nf[i][n])
	  resc[i][n] = corr[i][n] / nf[i][n];

    if (on_iteration != NULL) {
      double **snapshot;

      check_alloc(snapshot = (double **)malloc(sizeof(double *) * comp));
      for (i = 0; i < comp; i++) {
	check_alloc(snapshot[i] = (double *)malloc(sizeof(double) * length));
	for (n = 0; n < length; n++)
	  snapshot[i][n] = resc[i][n] * cinterval[i] + cmin[i];
      }
      on_iteration(iter, iterations, (double *const *)snapshot, nmf,
		   on_iteration_data);
      for (i = 0; i < comp; i++)
	free(snapshot[i]);
      free(snapshot);
    }
  }

  free(hcor);
  for (i = 0; i < comp; i++) {
    free(nf[i]);
    free(corr[i]);
  }
  free(nf);
  free(corr);
  for (n = 0; n < NRLAZY_BOX; n++)
    free(box[n]);
  free(box);
  free(list);
  for (i = 0; i < 2; i++)
    free(indexes[i]);
  free(indexes);

  check_alloc(result = (NRLazyResult *)malloc(sizeof(NRLazyResult)));
  result->comp = comp;
  result->length = length;
  check_alloc(result->series = (double **)malloc(sizeof(double *) * comp));
  for (i = 0; i < comp; i++) {
    check_alloc(result->series[i] = (double *)malloc(sizeof(double) * length));
    for (n = 0; n < length; n++)
      result->series[i][n] = resc[i][n] * cinterval[i] + cmin[i];
    free(resc[i]);
  }
  free(resc);
  free(cmin);
  free(cinterval);
  result->neighbors = nmf;

  return result;
}

void nrlazy_free(NRLazyResult *result)
{
  unsigned int i;

  if (result == NULL)
    return;
  for (i = 0; i < result->comp; i++)
    free(result->series[i]);
  free(result->series);
  free(result->neighbors);
  free(result);
}
