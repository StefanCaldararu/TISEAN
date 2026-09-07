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

/* Reentrant core of false_nearest, factored out of source_c/false_nearest.c
   so it has no dependency on argv parsing, file-scope globals, or the
   process-exiting error paths in the generic variance()/rescale_data()
   library routines it used to call. The math here (the box-assisted
   nearest-neighbor search in mmb()/find_nearest() and the per-embedding
   epsilon growth loop in main()) is unchanged from the original. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../routines/tsa.h"
#include "../../include/false_nearest.h"

#define BOX 1024
#define IBOX (BOX-1)

typedef struct {
  double *const *series;
  unsigned long length;
  unsigned int comp;
  unsigned int delay;
  unsigned int maxemb;
  unsigned long theiler;
  double rt;
  double varianz;
  long **box;
  long *list;
  unsigned int *vcomp;
  unsigned int *vemb;
  double aveps;
  double vareps;
  unsigned long toolarge;
} FNContext;

static void mmb(FNContext *ctx, unsigned int hdim, unsigned int hemb, double eps)
{
  unsigned long i;
  long x, y;
  long **box = ctx->box;
  long *list = ctx->list;
  double *const *series = ctx->series;

  for (x = 0; x < BOX; x++)
    for (y = 0; y < BOX; y++)
      box[x][y] = -1;

  for (i = 0; i < ctx->length - (ctx->maxemb + 1) * ctx->delay; i++) {
    x = (long)(series[0][i] / eps) & IBOX;
    y = (long)(series[hdim][i + hemb] / eps) & IBOX;
    list[i] = box[x][y];
    box[x][y] = i;
  }
}

static char find_nearest(FNContext *ctx, long n, unsigned int dim, double eps)
{
  long x, y, x1, x2, y1, i, i1, ic, ie;
  long element, which = -1;
  double dx, maxdx, mindx = 1.1, hfactor, factor;
  double *const *series = ctx->series;
  long **box = ctx->box;
  long *list = ctx->list;
  unsigned int *vcomp = ctx->vcomp;
  unsigned int *vemb = ctx->vemb;

  ic = vcomp[dim];
  ie = vemb[dim];
  x = (long)(series[0][n] / eps) & IBOX;
  y = (long)(series[ic][n + ie] / eps) & IBOX;

  for (x1 = x - 1; x1 <= x + 1; x1++) {
    x2 = x1 & IBOX;
    for (y1 = y - 1; y1 <= y + 1; y1++) {
      element = box[x2][y1 & IBOX];
      while (element != -1) {
	if ((unsigned long)labs(element - n) > ctx->theiler) {
	  maxdx = fabs(series[0][n] - series[0][element]);
	  for (i = 1; i <= dim; i++) {
	    ic = vcomp[i];
	    i1 = vemb[i];
	    dx = fabs(series[ic][n + i1] - series[ic][element + i1]);
	    if (dx > maxdx)
	      maxdx = dx;
	  }
	  if ((maxdx < mindx) && (maxdx > 0.0)) {
	    which = element;
	    mindx = maxdx;
	  }
	}
	element = list[element];
      }
    }
  }

  if ((which != -1) && (mindx <= eps) && (mindx <= ctx->varianz / ctx->rt)) {
    ctx->aveps += mindx;
    ctx->vareps += mindx * mindx;
    factor = 0.0;
    for (i = 1; i <= ctx->comp; i++) {
      ic = vcomp[dim + i];
      ie = vemb[dim + i];
      hfactor = fabs(series[ic][n + ie] - series[ic][which + ie]) / mindx;
      if (hfactor > factor)
	factor = hfactor;
    }
    if (factor > ctx->rt)
      ctx->toolarge++;
    return 1;
  }
  return 0;
}

static void free_series(double **series, unsigned int comp)
{
  unsigned int i;

  for (i = 0; i < comp; i++)
    free(series[i]);
  free(series);
}

FalseNearest *false_nearest_compute(double *const *series_in, unsigned long length,
				     unsigned int comp, unsigned int delay,
				     unsigned int minemb, unsigned int maxemb,
				     unsigned long theiler, double rt,
				     double eps0, FalseNearestError *error)
{
  unsigned long i, j, n;
  unsigned int c, emb, maxdim;
  long dim;
  double **series;
  double min, interval, average, var, varianz = 0.0, inter = 0.0, h;
  char *nearest;
  long **box;
  long *list;
  unsigned int *vcomp, *vemb;
  double epsilon;
  unsigned long donesofar;
  char alldone;
  FNContext ctx;
  FalseNearest *result;

  if (error != NULL)
    *error = FALSE_NEAREST_OK;

  check_alloc(series = (double **)malloc(sizeof(double *) * comp));
  for (c = 0; c < comp; c++) {
    check_alloc(series[c] = (double *)malloc(sizeof(double) * length));
    memcpy(series[c], series_in[c], sizeof(double) * length);
  }

  for (c = 0; c < comp; c++) {
    /* rescale_data(series[c], length, &min, &interval), on a private copy */
    min = interval = series[c][0];
    for (i = 1; i < length; i++) {
      if (series[c][i] < min) min = series[c][i];
      if (series[c][i] > interval) interval = series[c][i];
    }
    interval -= min;
    if (interval == 0.0) {
      free_series(series, comp);
      if (error != NULL)
	*error = FALSE_NEAREST_ERR_ZERO_INTERVAL;
      return NULL;
    }
    for (i = 0; i < length; i++)
      series[c][i] = (series[c][i] - min) / interval;

    /* variance(series[c], length, &average, &var), on the rescaled data */
    average = var = 0.0;
    for (i = 0; i < length; i++) {
      h = series[c][i];
      average += h;
      var += h * h;
    }
    average /= (double)length;
    var = sqrt(fabs(var / (double)length - average * average));
    if (var == 0.0) {
      free_series(series, comp);
      if (error != NULL)
	*error = FALSE_NEAREST_ERR_ZERO_VARIANCE;
      return NULL;
    }

    if (c == 0) {
      varianz = var;
      inter = interval;
    }
    else {
      varianz = (varianz > var) ? var : varianz;
      inter = (inter < interval) ? interval : inter;
    }
  }

  maxdim = comp * (maxemb + 1);

  check_alloc(list = (long *)malloc(sizeof(long) * length));
  check_alloc(nearest = (char *)malloc(length));
  check_alloc(box = (long **)malloc(sizeof(long *) * BOX));
  for (i = 0; i < BOX; i++)
    check_alloc(box[i] = (long *)malloc(sizeof(long) * BOX));

  check_alloc(vcomp = (unsigned int *)malloc(sizeof(unsigned int) * maxdim));
  check_alloc(vemb = (unsigned int *)malloc(sizeof(unsigned int) * maxdim));
  for (i = 0; i < maxdim; i++) {
    if (comp == 1) {
      vcomp[i] = 0;
      vemb[i] = i;
    }
    else {
      vcomp[i] = i % comp;
      vemb[i] = (i / comp) * delay;
    }
  }

  ctx.series = series;
  ctx.length = length;
  ctx.comp = comp;
  ctx.delay = delay;
  ctx.maxemb = maxemb;
  ctx.theiler = theiler;
  ctx.rt = rt;
  ctx.varianz = varianz;
  ctx.box = box;
  ctx.list = list;
  ctx.vcomp = vcomp;
  ctx.vemb = vemb;

  n = (minemb <= maxemb) ? (unsigned long)(maxemb - minemb) + 1 : 0;
  check_alloc(result = (FalseNearest *)malloc(sizeof(FalseNearest)));
  result->n = n;
  result->dimension = NULL;
  result->fraction = NULL;
  result->avg_eps = NULL;
  result->sigma_eps = NULL;
  if (n > 0) {
    check_alloc(result->dimension = (unsigned int *)malloc(sizeof(unsigned int) * n));
    check_alloc(result->fraction = (double *)malloc(sizeof(double) * n));
    check_alloc(result->avg_eps = (double *)malloc(sizeof(double) * n));
    check_alloc(result->sigma_eps = (double *)malloc(sizeof(double) * n));
  }

  j = 0;
  for (emb = minemb; emb <= maxemb; emb++) {
    dim = (long)emb * comp - 1;
    epsilon = eps0;
    ctx.toolarge = 0;
    alldone = 0;
    donesofar = 0;
    ctx.aveps = 0.0;
    ctx.vareps = 0.0;
    for (i = 0; i < length; i++)
      nearest[i] = 0;
    while (!alldone && (epsilon < 2. * varianz / rt)) {
      alldone = 1;
      mmb(&ctx, vcomp[dim], vemb[dim], epsilon);
      for (i = 0; i < length - maxemb * delay; i++)
	if (!nearest[i]) {
	  nearest[i] = find_nearest(&ctx, i, (unsigned int)dim, epsilon);
	  alldone &= nearest[i];
	  donesofar += (unsigned long)nearest[i];
	}
      epsilon *= sqrt(2.0);
      if (!donesofar)
	eps0 = epsilon;
    }
    if (donesofar == 0) {
      false_nearest_free(result);
      free(nearest);
      free(list);
      for (i = 0; i < BOX; i++)
	free(box[i]);
      free(box);
      free(vcomp);
      free(vemb);
      free_series(series, comp);
      if (error != NULL)
	*error = FALSE_NEAREST_ERR_NOT_ENOUGH_POINTS;
      return NULL;
    }
    result->dimension[j] = (unsigned int)(dim + 1);
    result->fraction[j] = (double)ctx.toolarge / (double)donesofar;
    result->avg_eps[j] = (ctx.aveps / (double)donesofar) * inter;
    result->sigma_eps[j] = sqrt(ctx.vareps / (double)donesofar) * inter;
    j++;
  }

  free(nearest);
  free(list);
  for (i = 0; i < BOX; i++)
    free(box[i]);
  free(box);
  free(vcomp);
  free(vemb);
  free_series(series, comp);

  return result;
}

void false_nearest_free(FalseNearest *fn)
{
  if (fn == NULL)
    return;
  free(fn->dimension);
  free(fn->fraction);
  free(fn->avg_eps);
  free(fn->sigma_eps);
  free(fn);
}
