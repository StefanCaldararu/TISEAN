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

/* Reentrant core of lzo-run, factored out of source_c/lzo-run.c so it has no
   dependency on argv parsing, file-scope globals, or the process-exiting
   error paths in the generic variance()/rescale_data() library routines it
   used to call. The math here (the per-component rescale/variance, the
   box-assisted neighbor search in put_in_boxes()/hfind_neighbors()/sort()
   and the zeroth-order fit in make_zeroth()/main()) is unchanged from the
   original. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../routines/tsa.h"
#include "../../include/lzo-run.h"

/* number of boxes for the neighbor search algorithm - must be a power of
   two, since box indices are formed with "& (NMAX - 1)" instead of "% NMAX" */
#define NMAX 128

/* Bundles what put_in_boxes()/hfind_neighbors()/sort()/make_zeroth() used to
   reach through file-scope globals, so those functions can stay reentrant. */
typedef struct {
  double **series;    /* [dim][length], rescaled to [0,1) */
  unsigned int dim, dim1, embed, delay;
  unsigned long length;
  unsigned int minn;
  long **box;          /* [NMAX][NMAX] */
  long *list;            /* [length] */
  long *found;            /* [length] */
  double *abstand;         /* [length] */
  unsigned int **indexes;   /* from make_multi_index() */
  double *var;               /* [dim], variance of the rescaled data */
  double q;                   /* noise_pct/100, only meaningful if setnoise */
  char setnoise;
  double epsilon;               /* current search radius */
  double **cast;                 /* [hdim+1][dim] ring buffer of the most
				     recent (embed-1)*delay+1 points */
  unsigned long hdim;             /* (embed-1)*delay */
} LzoRunCtx;

static void lzo_run_sort(LzoRunCtx *ctx, unsigned long nfound)
{
  double dx, dswap;
  long i, j, k, hf, iswap;
  long hdim = (long)ctx->hdim;

  for (i = 0; i < (long)nfound; i++) {
    hf = ctx->found[i];
    ctx->abstand[i] = 0.0;
    for (j = 0; j < (long)ctx->dim; j++) {
      for (k = 0; k <= hdim; k += ctx->delay) {
	dx = fabs(ctx->series[j][hf - k] - ctx->cast[hdim - k][j]);
	if (dx > ctx->abstand[i]) ctx->abstand[i] = dx;
      }
    }
  }

  for (i = 0; i < (long)ctx->minn; i++)
    for (j = i + 1; j < (long)nfound; j++)
      if (ctx->abstand[j] < ctx->abstand[i]) {
	dswap = ctx->abstand[i];
	ctx->abstand[i] = ctx->abstand[j];
	ctx->abstand[j] = dswap;
	iswap = ctx->found[i];
	ctx->found[i] = ctx->found[j];
	ctx->found[j] = iswap;
      }
}

static void lzo_run_put_in_boxes(LzoRunCtx *ctx)
{
  long i, j, n;
  long hdim = (long)ctx->hdim;
  double epsinv;

  epsinv = 1.0 / ctx->epsilon;
  for (i = 0; i < NMAX; i++)
    for (j = 0; j < NMAX; j++)
      ctx->box[i][j] = -1;

  for (n = hdim; n < (long)ctx->length - 1; n++) {
    i = (long)(ctx->series[0][n] * epsinv) & (NMAX - 1);
    j = (long)(ctx->series[ctx->dim1][n - hdim] * epsinv) & (NMAX - 1);
    ctx->list[n] = ctx->box[i][j];
    ctx->box[i][j] = n;
  }
}

static unsigned int lzo_run_hfind_neighbors(LzoRunCtx *ctx)
{
  char toolarge;
  long i, j, i1, i2, j1, l, hc, hd, element;
  long hdim = (long)ctx->hdim;
  unsigned int nfound = 0;
  double max, dx, epsinv;

  epsinv = 1.0 / ctx->epsilon;
  i = (long)(ctx->cast[hdim][0] * epsinv) & (NMAX - 1);
  j = (long)(ctx->cast[0][ctx->dim1] * epsinv) & (NMAX - 1);

  for (i1 = i - 1; i1 <= i + 1; i1++) {
    i2 = i1 & (NMAX - 1);
    for (j1 = j - 1; j1 <= j + 1; j1++) {
      element = ctx->box[i2][j1 & (NMAX - 1)];
      while (element != -1) {
	max = 0.0;
	toolarge = 0;
	for (l = 0; l < (long)(ctx->dim * ctx->embed); l++) {
	  hc = ctx->indexes[0][l];
	  hd = ctx->indexes[1][l];
	  dx = fabs(ctx->series[hc][element - hd] - ctx->cast[hdim - hd][hc]);
	  max = (dx > max) ? dx : max;
	  if (max > ctx->epsilon) {
	    toolarge = 1;
	    break;
	  }
	}
	if (max <= ctx->epsilon)
	  ctx->found[nfound++] = element;
	element = ctx->list[element];
      }
    }
  }
  (void)toolarge;
  return nfound;
}

static void lzo_run_make_zeroth(LzoRunCtx *ctx, long number, double *newcast)
{
  long d, i;
  double *sd;

  for (d = 0; d < (long)ctx->dim; d++) {
    newcast[d] = 0.0;
    sd = ctx->series[d] + 1;
    for (i = 0; i < number; i++)
      newcast[d] += sd[ctx->found[i]];
    newcast[d] /= (double)number;
  }

  if (ctx->setnoise) {
    for (d = 0; d < (long)ctx->dim; d++)
      newcast[d] += gaussian(ctx->var[d] * ctx->q);
  }
}

LzoRun *lzo_run_forecast(double *const *series_in, unsigned long length,
			   unsigned int dim, unsigned int embed,
			   unsigned int delay, unsigned int minn,
			   char fix_neighbors, unsigned long flength,
			   double eps0, char epsset, double epsf,
			   double noise_pct, char setnoise,
			   unsigned long seed, LzoRunError *error)
{
  unsigned long i, k, count, cast_size;
  long j, actfound;
  double h, average, var_h, min_h, max_h;
  double *min, *interval, *var, *newcast, *swap;
  double maxinterval, epsilon0, q;
  double **series, **cast, **out;
  char done;
  LzoRunCtx ctx;
  LzoRun *result;

  if (error != NULL)
    *error = LZO_RUN_OK;

  check_alloc(series = (double **)malloc(sizeof(double *) * dim));
  for (i = 0; i < dim; i++) {
    check_alloc(series[i] = (double *)malloc(sizeof(double) * length));
    memcpy(series[i], series_in[i], sizeof(double) * length);
  }

  check_alloc(min = (double *)malloc(sizeof(double) * dim));
  check_alloc(interval = (double *)malloc(sizeof(double) * dim));
  check_alloc(var = (double *)malloc(sizeof(double) * dim));

  maxinterval = 0.0;

  for (i = 0; i < dim; i++) {
    /* rescale_data(series[i], length, &min[i], &interval[i]), on our own
       private copy */
    min_h = max_h = series[i][0];
    for (k = 1; k < length; k++) {
      if (series[i][k] < min_h) min_h = series[i][k];
      if (series[i][k] > max_h) max_h = series[i][k];
    }
    max_h -= min_h;
    if (max_h == 0.0) {
      for (k = 0; k < dim; k++)
	free(series[k]);
      free(series);
      free(min);
      free(interval);
      free(var);
      if (error != NULL)
	*error = LZO_RUN_ERR_ZERO_INTERVAL;
      return NULL;
    }
    for (k = 0; k < length; k++)
      series[i][k] = (series[i][k] - min_h) / max_h;
    min[i] = min_h;
    interval[i] = max_h;

    /* variance(series[i], length, &dummy, &var[i]), on the rescaled data */
    average = var_h = 0.0;
    for (k = 0; k < length; k++) {
      h = series[i][k];
      average += h;
      var_h += h * h;
    }
    average /= (double)length;
    var_h = sqrt(fabs(var_h / (double)length - average * average));
    if (var_h == 0.0) {
      for (k = 0; k < dim; k++)
	free(series[k]);
      free(series);
      free(min);
      free(interval);
      free(var);
      if (error != NULL)
	*error = LZO_RUN_ERR_ZERO_VARIANCE;
      return NULL;
    }
    var[i] = var_h;

    if (interval[i] > maxinterval)
      maxinterval = interval[i];
  }

  if (epsset)
    eps0 /= maxinterval;

  ctx.hdim = (unsigned long)(embed - 1) * delay;
  cast_size = ctx.hdim + 1;

  check_alloc(cast = (double **)malloc(sizeof(double *) * cast_size));
  for (i = 0; i < cast_size; i++)
    check_alloc(cast[i] = (double *)malloc(sizeof(double) * dim));
  check_alloc(newcast = (double *)malloc(sizeof(double) * dim));

  check_alloc(ctx.list = (long *)malloc(sizeof(long) * length));
  check_alloc(ctx.found = (long *)malloc(sizeof(long) * length));
  check_alloc(ctx.abstand = (double *)malloc(sizeof(double) * length));
  check_alloc(ctx.box = (long **)malloc(sizeof(long *) * NMAX));
  for (i = 0; i < NMAX; i++)
    check_alloc(ctx.box[i] = (long *)malloc(sizeof(long) * NMAX));

  for (j = 0; j < (long)dim; j++)
    for (i = 0; i < cast_size; i++)
      cast[i][j] = series[j][length - cast_size + i];

  ctx.indexes = make_multi_index(dim, embed, delay);

  ctx.series = series;
  ctx.dim = dim;
  ctx.dim1 = dim - 1;
  ctx.embed = embed;
  ctx.delay = delay;
  ctx.length = length;
  ctx.minn = minn;
  ctx.var = var;
  ctx.setnoise = setnoise;
  ctx.cast = cast;
  q = setnoise ? noise_pct / 100.0 : 0.0;
  ctx.q = q;

  rnd_init(seed);

  epsilon0 = eps0 / epsf;
  count = 1;

  check_alloc(out = (double **)malloc(sizeof(double *) * flength));

  for (i = 0; i < flength; i++) {
    done = 0;
    if (fix_neighbors)
      ctx.epsilon = epsilon0 / ((double)count * epsf);
    else
      ctx.epsilon = epsilon0;
    while (!done) {
      ctx.epsilon *= epsf;
      lzo_run_put_in_boxes(&ctx);
      actfound = (long)lzo_run_hfind_neighbors(&ctx);
      if (actfound >= (long)minn) {
	if (fix_neighbors) {
	  epsilon0 += ctx.epsilon;
	  count++;
	  lzo_run_sort(&ctx, (unsigned long)actfound);
	  actfound = (long)minn;
	}
	lzo_run_make_zeroth(&ctx, actfound, newcast);
	check_alloc(out[i] = (double *)malloc(sizeof(double) * dim));
	for (j = 0; j < (long)dim; j++)
	  out[i][j] = newcast[j] * interval[j] + min[j];
	done = 1;
	swap = cast[0];
	for (j = 0; j < (long)cast_size - 1; j++)
	  cast[j] = cast[j + 1];
	cast[cast_size - 1] = swap;
	for (j = 0; j < (long)dim; j++)
	  cast[cast_size - 1][j] = newcast[j];
      }
    }
  }

  for (i = 0; i < dim; i++)
    free(series[i]);
  free(series);
  free(min);
  free(interval);
  free(var);
  for (i = 0; i < cast_size; i++)
    free(cast[i]);
  free(cast);
  free(newcast);
  free(ctx.list);
  free(ctx.found);
  free(ctx.abstand);
  for (i = 0; i < NMAX; i++)
    free(ctx.box[i]);
  free(ctx.box);
  for (i = 0; i < 2; i++)
    free(ctx.indexes[i]);
  free(ctx.indexes);

  check_alloc(result = (LzoRun *)malloc(sizeof(LzoRun)));
  result->dim = dim;
  result->length = flength;
  check_alloc(result->series = (double *)malloc(sizeof(double) * flength * dim));
  for (i = 0; i < flength; i++) {
    for (j = 0; j < (long)dim; j++)
      result->series[i * dim + j] = out[i][j];
    free(out[i]);
  }
  free(out);

  return result;
}

void lzo_run_free(LzoRun *result)
{
  if (result == NULL)
    return;
  free(result->series);
  free(result);
}
