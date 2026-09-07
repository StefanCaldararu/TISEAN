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

/* Reentrant core of lzo-test, factored out of source_c/lzo-test.c so it has
   no dependency on argv parsing, file-scope globals, or the process-exiting
   error paths in the generic variance()/rescale_data() library routines it
   used to call. The math here (the per-component rescale/variance, the
   box-assisted multi-dimensional neighbor search and the zeroth-order fit
   in make_fit()/main()) is unchanged from the original. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../routines/tsa.h"
#include "../../include/lzo-test.h"

/*number of boxes for the neighbor search algorithm*/
#define NMAX 512

static void make_fit(double *const *series, const unsigned long *found,
		      unsigned long number, long act, long istep,
		      unsigned int dim, double **diffs, double **error)
{
  double casted, *help;
  unsigned int j;
  unsigned long i;
  long h = istep - 1;

  for (j = 0; j < dim; j++) {
    casted = 0.0;
    help = series[j] + istep;
    for (i = 0; i < number; i++)
      casted += help[found[i]];
    casted /= (double)number;
    diffs[j][act] = casted - help[act];
    error[j][h] += sqr(casted - help[act]);
  }
}

LzoTest *lzo_test_compute(double *const *series_in, unsigned long length,
			   unsigned int dim, unsigned int embed,
			   unsigned int delay, unsigned int minn,
			   unsigned long step, unsigned long refstep,
			   unsigned long causal, unsigned long clength_opt,
			   char clengthset, double eps0, char epsset,
			   double epsf, LzoTestError *error)
{
  unsigned long i, j, lowbound, clength, n_ref, idx;
  long hi;
  double **series, *av, *rms, *hinter, **hser;
  double min, comp_interval, avg_interval, epsilon, h, average, var;
  long **box, *list;
  unsigned long *found, *hfound, actfound;
  char *done, alldone;
  double **diffs, **err, *result_error, *result_diffs;
  LzoTest *result;

  if (error != NULL)
    *error = LZO_TEST_OK;

  check_alloc(series = (double **)malloc(sizeof(double *) * dim));
  for (j = 0; j < dim; j++) {
    check_alloc(series[j] = (double *)malloc(sizeof(double) * length));
    memcpy(series[j], series_in[j], sizeof(double) * length);
  }

  check_alloc(av = (double *)malloc(sizeof(double) * dim));
  check_alloc(rms = (double *)malloc(sizeof(double) * dim));
  check_alloc(hinter = (double *)malloc(sizeof(double) * dim));
  avg_interval = 0.0;
  for (j = 0; j < dim; j++) {
    /* rescale_data(series[j], length, &min, &hinter[j]), on a private copy */
    min = comp_interval = series[j][0];
    for (i = 1; i < length; i++) {
      if (series[j][i] < min) min = series[j][i];
      if (series[j][i] > comp_interval) comp_interval = series[j][i];
    }
    comp_interval -= min;
    if (comp_interval == 0.0) {
      for (i = 0; i < dim; i++)
	free(series[i]);
      free(series);
      free(av);
      free(rms);
      free(hinter);
      if (error != NULL)
	*error = LZO_TEST_ERR_ZERO_INTERVAL;
      return NULL;
    }
    for (i = 0; i < length; i++)
      series[j][i] = (series[j][i] - min) / comp_interval;

    /* variance(series[j], length, &av[j], &rms[j]), on the rescaled data */
    average = var = 0.0;
    for (i = 0; i < length; i++) {
      h = series[j][i];
      average += h;
      var += h * h;
    }
    average /= (double)length;
    var = sqrt(fabs(var / (double)length - average * average));
    if (var == 0.0) {
      for (i = 0; i < dim; i++)
	free(series[i]);
      free(series);
      free(av);
      free(rms);
      free(hinter);
      if (error != NULL)
	*error = LZO_TEST_ERR_ZERO_VARIANCE;
      return NULL;
    }
    av[j] = average;
    rms[j] = var;
    hinter[j] = comp_interval;
    avg_interval += comp_interval;
  }
  avg_interval /= (double)dim;

  check_alloc(hser = (double **)malloc(sizeof(double *) * dim));
  check_alloc(list = (long *)malloc(sizeof(long) * length));
  check_alloc(found = (unsigned long *)malloc(sizeof(long) * length));
  check_alloc(hfound = (unsigned long *)malloc(sizeof(long) * length));
  check_alloc(done = (char *)malloc(sizeof(char) * length));
  check_alloc(box = (long **)malloc(sizeof(long *) * NMAX));
  check_alloc(err = (double **)malloc(sizeof(double *) * dim));
  check_alloc(diffs = (double **)malloc(sizeof(double *) * dim));
  for (j = 0; j < dim; j++) {
    check_alloc(diffs[j] = (double *)malloc(sizeof(double) * length));
    check_alloc(err[j] = (double *)malloc(sizeof(double) * step));
    for (i = 0; i < step; i++)
      err[j][i] = 0.0;
  }

  for (i = 0; i < NMAX; i++)
    check_alloc(box[i] = (long *)malloc(sizeof(long) * NMAX));

  for (i = 0; i < length; i++)
    done[i] = 0;

  alldone = 0;
  if (epsset)
    eps0 /= avg_interval;

  epsilon = eps0 / epsf;

  clength = clengthset ? clength_opt : length;
  clength = ((clength * refstep + step) <= length) ? clength :
    (length - (long)step) / refstep;

  while (!alldone) {
    alldone = 1;
    epsilon *= epsf;
    make_multi_box(series, box, list, length - (long)step, NMAX, dim, embed,
		    delay, epsilon);
    for (i = (embed - 1) * delay; i < clength; i++)
      if (!done[i]) {
	hi = i * refstep;
	for (j = 0; j < dim; j++)
	  hser[j] = series[j] + hi;
	actfound = find_multi_neighbors(series, box, list, hser, length, NMAX,
					 dim, embed, delay, epsilon, hfound);
	actfound = exclude_interval(actfound, hi - (long)causal + 1,
				     hi + causal + (embed - 1) * delay - 1,
				     hfound, found);
	if (actfound >= minn) {
	  for (j = 1; j <= step; j++)
	    make_fit(series, found, actfound, hi, (long)j, dim, diffs, err);
	  done[i] = 1;
	}
	alldone &= done[i];
      }
  }

  check_alloc(result_error = (double *)malloc(sizeof(double) * step * dim));
  for (i = 0; i < step; i++)
    for (j = 0; j < dim; j++)
      result_error[i * dim + j] =
	  sqrt(err[j][i] / (clength - (embed - 1) * delay)) / rms[j];

  lowbound = (embed - 1) * delay;
  n_ref = 0;
  for (i = lowbound; i < clength; i++)
    n_ref++;
  result_diffs = NULL;
  if (n_ref > 0)
    check_alloc(result_diffs = (double *)malloc(sizeof(double) * n_ref * dim));
  idx = 0;
  for (i = lowbound; i < clength; i++) {
    hi = i * refstep;
    for (j = 0; j < dim; j++)
      result_diffs[idx * dim + j] = diffs[j][hi] * hinter[j];
    idx++;
  }

  free(list);
  free(found);
  free(hfound);
  free(done);
  for (i = 0; i < NMAX; i++)
    free(box[i]);
  free(box);
  free(hser);
  for (j = 0; j < dim; j++) {
    free(series[j]);
    free(diffs[j]);
    free(err[j]);
  }
  free(series);
  free(diffs);
  free(err);
  free(av);
  free(rms);
  free(hinter);

  check_alloc(result = (LzoTest *)malloc(sizeof(LzoTest)));
  result->dim = dim;
  result->step = step;
  result->error = result_error;
  result->n_ref = n_ref;
  result->diffs = result_diffs;

  return result;
}

void lzo_test_free(LzoTest *result)
{
  if (result == NULL)
    return;
  free(result->error);
  free(result->diffs);
  free(result);
}
