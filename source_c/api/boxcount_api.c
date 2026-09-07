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

/* Reentrant core of boxcount, factored out of source_c/boxcount.c so it has
   no dependency on argv parsing, file-scope globals, or the process-exiting
   error path in the generic rescale_data() library routine it used to call.
   The math here (start_box()/next_dim()'s recursive box partition and the
   epsilon-stepping loop in main()) is unchanged from the original. */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../routines/tsa.h"
#include "../../include/boxcount.h"

/* Replaces the global series/length/epsi/histo/which_dims/delay/q that
   start_box()/next_dim() used to close over, so the recursion is reentrant. */
typedef struct {
  double **series;             /* [dimension][length], rescaled in place */
  unsigned long length;        /* reduced length: length - (maxembed-1)*delay */
  int epsi;                    /* number of boxes for the current epsilon */
  unsigned int delay;
  double q;
  double *histo;               /* [dimension*maxembed] accumulator */
  unsigned int **which_dims;   /* [dimension*maxembed][2]: {component,embed} */
  unsigned int maxembed;
  unsigned int dimension;
} BoxWorkspace;

static void next_dim_core(BoxWorkspace *ws, int wd, int n, unsigned int *first)
{
  int i, which, d1, comp;
  double epsinv, norm, p;
  unsigned int **act;
  int *found, hf;

  comp = ws->which_dims[wd][0];
  d1 = ws->which_dims[wd][1] * ws->delay;

  epsinv = (double)ws->epsi;
  norm = (double)ws->length;

  check_alloc(act = (unsigned int **)malloc(ws->epsi * sizeof(int *)));
  check_alloc(found = (int *)malloc(ws->epsi * sizeof(int)));

  for (i = 0; i < ws->epsi; i++) {
    found[i] = 0;
    act[i] = NULL;
  }

  for (i = 0; i < n; i++) {
    which = (int)(ws->series[comp][first[i] + d1] * epsinv);
    hf = ++found[which];
    check_alloc(act[which] =
		realloc((unsigned int *)act[which], hf * sizeof(unsigned int)));
    act[which][hf - 1] = first[i];
  }

  for (i = 0; i < ws->epsi; i++)
    if (found[i]) {
      p = (double)(found[i]) / (norm);
      if (ws->q == 1.0)
	ws->histo[wd] -= p * log(p);
      else
	ws->histo[wd] += pow(p, ws->q);
    }

  if (wd < (int)(ws->maxembed * ws->dimension) - 1)
    for (i = 0; i < ws->epsi; i++)
      if (found[i])
	next_dim_core(ws, wd + 1, found[i], act[i]);

  for (i = 0; i < ws->epsi; i++)
    if (found[i])
      free(act[i]);

  free(act);
  free(found);
}

static void start_box_core(BoxWorkspace *ws)
{
  int i, which;
  double epsinv, norm, p;
  unsigned int **act;
  int *found, hf;

  epsinv = (double)ws->epsi;
  norm = (double)ws->length;

  check_alloc(act = (unsigned int **)malloc(ws->epsi * sizeof(int *)));
  check_alloc(found = (int *)malloc(ws->epsi * sizeof(int)));

  for (i = 0; i < ws->epsi; i++) {
    found[i] = 0;
    act[i] = NULL;
  }

  for (i = 0; (unsigned long)i < ws->length; i++) {
    which = (int)(ws->series[0][i] * epsinv);
    hf = ++found[which];
    check_alloc(act[which] =
		realloc((unsigned int *)act[which], hf * sizeof(unsigned int)));
    act[which][hf - 1] = i;
  }

  for (i = 0; i < ws->epsi; i++)
    if (found[i]) {
      p = (double)(found[i]) / (norm);
      if (ws->q == 1.0)
	ws->histo[0] -= p * log(p);
      else
	ws->histo[0] += pow(p, ws->q);
    }

  if (1 < (int)(ws->dimension * ws->maxembed)) {
    for (i = 0; i < ws->epsi; i++) {
      if (found[i])
	next_dim_core(ws, 1, found[i], act[i]);
    }
  }

  for (i = 0; i < ws->epsi; i++)
    if (found[i])
      free(act[i]);

  free(act);
  free(found);
}

BoxCount *boxcount_compute(double *const *series_in, unsigned long length,
			    unsigned int dimension, unsigned int maxembed,
			    unsigned int delay, double q,
			    double epsmin, int epsmin_absolute,
			    double epsmax, int epsmax_absolute,
			    unsigned int epscount, BoxCountError *error)
{
  unsigned int i, j;
  unsigned long n, k;
  double **series;
  double min, interval, maxinterval;
  double *histo;
  unsigned int **which_dims;
  unsigned long reduced_length;
  double epsfaktor, heps;
  int epsi, epsi_old, epsi_test;
  BoxCount *bc;
  BoxWorkspace ws;

  if (error != NULL)
    *error = BOXCOUNT_OK;

  /* Private, rescaled copy of the input - boxcount_compute must not mutate
     the caller's data the way the CLI's in-place rescale_data()/EPSMIN-nudge
     loop does. */
  check_alloc(series = (double **)malloc(sizeof(double *) * dimension));
  for (i = 0; i < dimension; i++) {
    check_alloc(series[i] = (double *)malloc(sizeof(double) * length));
    for (n = 0; n < length; n++)
      series[i][n] = series_in[i][n];
  }

  maxinterval = 0.0;
  for (i = 0; i < dimension; i++) {
    /* rescale_data(series[i], length, &min, &interval), inlined so a
       zero-range component returns an error instead of exiting. */
    min = interval = series[i][0];
    for (n = 1; n < length; n++) {
      if (series[i][n] < min) min = series[i][n];
      if (series[i][n] > interval) interval = series[i][n];
    }
    interval -= min;

    if (interval == 0.0) {
      for (j = 0; j < dimension; j++)
	free(series[j]);
      free(series);
      if (error != NULL)
	*error = BOXCOUNT_ERR_ZERO_INTERVAL;
      return NULL;
    }

    for (n = 0; n < length; n++)
      series[i][n] = (series[i][n] - min) / interval;
    if (interval > maxinterval)
      maxinterval = interval;
  }

  if (epsmin_absolute)
    epsmin /= maxinterval;
  if (epsmax_absolute)
    epsmax /= maxinterval;

  for (i = 0; i < dimension; i++)
    for (n = 0; n < length; n++)
      if (series[i][n] >= 1.0)
	series[i][n] -= epsmin / 2.0;

  check_alloc(histo = (double *)malloc(sizeof(double) * maxembed * dimension));
  check_alloc(which_dims =
	      (unsigned int **)malloc(sizeof(unsigned int *) * maxembed * dimension));
  for (i = 0; i < maxembed * dimension; i++)
    check_alloc(which_dims[i] = (unsigned int *)malloc(sizeof(unsigned int) * 2));
  for (i = 0; i < maxembed; i++)
    for (j = 0; j < dimension; j++) {
      which_dims[i * dimension + j][0] = j;
      which_dims[i * dimension + j][1] = i;
    }

  reduced_length = length - (unsigned long)(maxembed - 1) * delay;

  if (epscount > 1)
    epsfaktor = pow(epsmax / epsmin, 1.0 / (double)(epscount - 1));
  else
    epsfaktor = 1.0;

  check_alloc(bc = (BoxCount *)malloc(sizeof(BoxCount)));
  bc->dimension = dimension;
  bc->maxembed = maxembed;
  bc->epscount = epscount;
  bc->eps = NULL;
  bc->entropy = NULL;
  bc->which_component = NULL;
  bc->which_embed = NULL;

  if (epscount > 0) {
    check_alloc(bc->eps = (double *)malloc(sizeof(double) * epscount));
    check_alloc(bc->entropy = (double **)malloc(sizeof(double *) * epscount));
    for (k = 0; k < epscount; k++)
      check_alloc(bc->entropy[k] =
		  (double *)malloc(sizeof(double) * maxembed * dimension));
  }
  check_alloc(bc->which_component =
	      (unsigned int *)malloc(sizeof(unsigned int) * maxembed * dimension));
  check_alloc(bc->which_embed =
	      (unsigned int *)malloc(sizeof(unsigned int) * maxembed * dimension));
  for (i = 0; i < maxembed * dimension; i++) {
    bc->which_component[i] = which_dims[i][0];
    bc->which_embed[i] = which_dims[i][1];
  }

  ws.series = series;
  ws.length = reduced_length;
  ws.delay = delay;
  ws.q = q;
  ws.histo = histo;
  ws.which_dims = which_dims;
  ws.maxembed = maxembed;
  ws.dimension = dimension;

  heps = epsmax * epsfaktor;
  epsi_old = 0;

  for (k = 0; k < epscount; k++) {
    for (i = 0; i < maxembed * dimension; i++)
      histo[i] = 0.0;

    do {
      heps /= epsfaktor;
      epsi_test = (int)(1. / heps);
    } while (epsi_test <= epsi_old);
    epsi = epsi_test;
    epsi_old = epsi;
    ws.epsi = epsi;

    start_box_core(&ws);

    for (i = 0; i < maxembed * dimension; i++) {
      if (q == 1.0)
	bc->entropy[k][i] = histo[i];
      else
	bc->entropy[k][i] = log(histo[i]) / (1.0 - q);
    }
    bc->eps[k] = heps * maxinterval;
  }

  for (i = 0; i < dimension; i++)
    free(series[i]);
  free(series);
  free(histo);
  for (i = 0; i < maxembed * dimension; i++)
    free(which_dims[i]);
  free(which_dims);

  return bc;
}

void boxcount_free(BoxCount *bc)
{
  unsigned long k;

  if (bc == NULL)
    return;
  if (bc->entropy != NULL) {
    for (k = 0; k < bc->epscount; k++)
      free(bc->entropy[k]);
    free(bc->entropy);
  }
  free(bc->eps);
  free(bc->which_component);
  free(bc->which_embed);
  free(bc);
}
