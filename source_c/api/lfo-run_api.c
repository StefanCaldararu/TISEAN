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

/* Reentrant core of lfo-run, factored out of source_c/lfo-run.c so it has no
   dependency on argv parsing, file-scope globals, or the process-exiting
   error path in the generic rescale_data() library routine it used to
   call. The math here (the per-component rescale, the box-assisted
   neighbor search in put_in_boxes()/hfind_neighbors(), the local-linear fit
   in multiply_matrix()/make_fit() and the zeroth order fit in
   make_zeroth()) is unchanged from the original. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../routines/tsa.h"
#include "../../include/lfo-run.h"

/* number of boxes for the neighbor search algorithm - must be a power of
   two, since box indices are formed with "& (NMAX - 1)" instead of "% NMAX" */
#define NMAX 128

/* Bundles what put_in_boxes()/hfind_neighbors()/make_fit()/make_zeroth()
   used to reach through file-scope globals, so those functions can stay
   reentrant. */
typedef struct {
  double **series;   /* [dim][length], rescaled to [0,1) */
  unsigned int dim, dim1, embed, delay;
  unsigned long length;
  unsigned int minn;
  long **box;          /* [NMAX][NMAX] */
  long *list;            /* [length] */
  long *found;            /* [length] */
  double epsilon;           /* current search radius */
  double **cast;              /* [hdim+1][dim] ring buffer of the most
				  recent (embed-1)*delay+1 points */
  unsigned long hdim;           /* (embed-1)*delay */
  double **mat;                   /* [dim*embed][dim*embed] scratch matrix */
  double *vec, *localav, *foreav;  /* scratch vectors for make_fit() */
} LfoRunCtx;

static void lfo_run_put_in_boxes(LfoRunCtx *ctx)
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

static unsigned int lfo_run_hfind_neighbors(LfoRunCtx *ctx)
{
  char toolarge;
  long i, j, i1, i2, j1, k, l, element;
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
	for (l = 0; l < (long)ctx->dim; l++) {
	  for (k = 0; k <= hdim; k += ctx->delay) {
	    dx = fabs(ctx->series[l][element - k] - ctx->cast[hdim - k][l]);
	    max = (dx > max) ? dx : max;
	    if (max > ctx->epsilon) {
	      toolarge = 1;
	      break;
	    }
	  }
	  if (toolarge)
	    break;
	}
	if (max <= ctx->epsilon)
	  ctx->found[nfound++] = element;
	element = ctx->list[element];
      }
    }
  }
  return nfound;
}

static void lfo_run_multiply_matrix(LfoRunCtx *ctx, double **mat, double *vec)
{
  double *hvec;
  long i, j;
  long n = (long)ctx->dim * (long)ctx->embed;

  check_alloc(hvec = (double *)malloc(sizeof(double) * n));
  for (i = 0; i < n; i++) {
    hvec[i] = 0.0;
    for (j = 0; j < n; j++)
      hvec[i] += mat[i][j] * vec[j];
  }
  for (i = 0; i < n; i++)
    vec[i] = hvec[i];
  free(hvec);
}

static void lfo_run_make_fit(LfoRunCtx *ctx, long number, double *newcast)
{
  double *sj, *si, lavi, lavj, fav;
  long i, i1, j, j1, hi, hj, hi1, hj1, n, which;
  long hdim = (long)ctx->hdim;
  long dim = (long)ctx->dim, embed = (long)ctx->embed, delay = (long)ctx->delay;
  double **imat;

  for (i = 0; i < dim * embed; i++)
    ctx->localav[i] = 0.0;
  for (i = 0; i < dim; i++)
    ctx->foreav[i] = 0.0;

  for (n = 0; n < number; n++) {
    which = ctx->found[n];
    for (j = 0; j < dim; j++) {
      sj = ctx->series[j];
      ctx->foreav[j] += sj[which + 1];
      for (j1 = 0; j1 < embed; j1++) {
	hj = j * embed + j1;
	ctx->localav[hj] += sj[which - j1 * delay];
      }
    }
  }

  for (i = 0; i < dim * embed; i++)
    ctx->localav[i] /= number;
  for (i = 0; i < dim; i++)
    ctx->foreav[i] /= number;

  for (i = 0; i < dim; i++) {
    si = ctx->series[i];
    for (i1 = 0; i1 < embed; i1++) {
      hi = i * embed + i1;
      lavi = ctx->localav[hi];
      hi1 = i1 * delay;
      for (j = 0; j < dim; j++) {
	sj = ctx->series[j];
	for (j1 = 0; j1 < embed; j1++) {
	  hj = j * embed + j1;
	  lavj = ctx->localav[hj];
	  hj1 = j1 * delay;
	  ctx->mat[hi][hj] = 0.0;
	  if (hj >= hi) {
	    for (n = 0; n < number; n++) {
	      which = ctx->found[n];
	      ctx->mat[hi][hj] += (si[which - hi1] - lavi) * (sj[which - hj1] - lavj);
	    }
	  }
	}
      }
    }
  }

  for (i = 0; i < dim * embed; i++)
    for (j = i; j < dim * embed; j++) {
      ctx->mat[i][j] /= number;
      ctx->mat[j][i] = ctx->mat[i][j];
    }

  imat = invert_matrix(ctx->mat, (unsigned int)(dim * embed));

  for (i = 0; i < dim; i++) {
    si = ctx->series[i];
    fav = ctx->foreav[i];
    for (j = 0; j < dim; j++) {
      sj = ctx->series[j];
      for (j1 = 0; j1 < embed; j1++) {
	hj = j * embed + j1;
	lavj = ctx->localav[hj];
	hj1 = j1 * delay;
	ctx->vec[hj] = 0.0;
	for (n = 0; n < number; n++) {
	  which = ctx->found[n];
	  ctx->vec[hj] += (si[which + 1] - fav) * (sj[which - hj1] - lavj);
	}
	ctx->vec[hj] /= number;
      }
    }

    lfo_run_multiply_matrix(ctx, imat, ctx->vec);

    newcast[i] = ctx->foreav[i];
    for (j = 0; j < dim; j++) {
      for (j1 = 0; j1 < embed; j1++) {
	hj = j * embed + j1;
	newcast[i] += ctx->vec[hj] * (ctx->cast[hdim - j1 * delay][j] - ctx->localav[hj]);
      }
    }
  }

  for (i = 0; i < dim * embed; i++)
    free(imat[i]);
  free(imat);
}

static void lfo_run_make_zeroth(LfoRunCtx *ctx, long number, double *newcast)
{
  unsigned long i, d;
  double *sj;

  for (d = 0; d < ctx->dim; d++) {
    newcast[d] = 0.0;
    sj = ctx->series[d] + 1;
    for (i = 0; i < (unsigned long)number; i++)
      newcast[d] += sj[ctx->found[i]];
    newcast[d] /= number;
  }
}

LfoRun *lfo_run_forecast(double *const *series_in, unsigned long length,
			  unsigned int dim, unsigned int embed,
			  unsigned int delay, unsigned int minn,
			  char do_zeroth, unsigned long flength,
			  double eps0, char epsset, double epsf,
			  LfoRunError *error)
{
  unsigned long i, k, cast_size, actual_count;
  long j, actfound;
  double h, average, var_h, min_h, max_h;
  double *min, *interval, *newcast, *swap;
  double maxinterval, epsilon;
  double **series, **cast, **out;
  char done, escaped;
  LfoRunCtx ctx;
  LfoRun *result;

  if (error != NULL)
    *error = LFO_RUN_OK;

  check_alloc(series = (double **)malloc(sizeof(double *) * dim));
  for (i = 0; i < dim; i++) {
    check_alloc(series[i] = (double *)malloc(sizeof(double) * length));
    memcpy(series[i], series_in[i], sizeof(double) * length);
  }

  check_alloc(min = (double *)malloc(sizeof(double) * dim));
  check_alloc(interval = (double *)malloc(sizeof(double) * dim));

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
      if (error != NULL)
	*error = LFO_RUN_ERR_ZERO_INTERVAL;
      return NULL;
    }
    for (k = 0; k < length; k++)
      series[i][k] = (series[i][k] - min_h) / max_h;
    min[i] = min_h;
    interval[i] = max_h;

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
  check_alloc(ctx.box = (long **)malloc(sizeof(long *) * NMAX));
  for (i = 0; i < NMAX; i++)
    check_alloc(ctx.box[i] = (long *)malloc(sizeof(long) * NMAX));

  check_alloc(ctx.localav = (double *)malloc(sizeof(double) * dim * embed));
  check_alloc(ctx.foreav = (double *)malloc(sizeof(double) * dim));
  check_alloc(ctx.vec = (double *)malloc(sizeof(double) * dim * embed));
  check_alloc(ctx.mat = (double **)malloc(sizeof(double *) * dim * embed));
  for (i = 0; i < (unsigned long)dim * embed; i++)
    check_alloc(ctx.mat[i] = (double *)malloc(sizeof(double) * dim * embed));

  for (j = 0; j < (long)dim; j++)
    for (i = 0; i < cast_size; i++)
      cast[i][j] = series[j][length - cast_size + i];

  ctx.series = series;
  ctx.dim = dim;
  ctx.dim1 = dim - 1;
  ctx.embed = embed;
  ctx.delay = delay;
  ctx.length = length;
  ctx.minn = minn;
  ctx.cast = cast;

  check_alloc(out = (double **)malloc(sizeof(double *) * flength));
  actual_count = 0;
  escaped = 0;

  for (i = 0; i < flength; i++) {
    done = 0;
    epsilon = eps0 / epsf;
    while (!done) {
      epsilon *= epsf;
      ctx.epsilon = epsilon;
      lfo_run_put_in_boxes(&ctx);
      actfound = (long)lfo_run_hfind_neighbors(&ctx);
      if (actfound >= (long)minn) {
	if (!do_zeroth)
	  lfo_run_make_fit(&ctx, actfound, newcast);
	else
	  lfo_run_make_zeroth(&ctx, actfound, newcast);

	check_alloc(out[i] = (double *)malloc(sizeof(double) * dim));
	for (j = 0; j < (long)dim; j++)
	  out[i][j] = newcast[j] * interval[j] + min[j];
	actual_count = i + 1;
	done = 1;

	for (j = 0; j < (long)dim; j++) {
	  if ((newcast[j] > 2.0) || (newcast[j] < -1.0))
	    escaped = 1;
	}
	if (escaped)
	  break;

	swap = cast[0];
	for (j = 0; j < (long)cast_size - 1; j++)
	  cast[j] = cast[j + 1];
	cast[cast_size - 1] = swap;
	for (j = 0; j < (long)dim; j++)
	  cast[cast_size - 1][j] = newcast[j];
      }
    }
    if (escaped)
      break;
  }

  for (i = 0; i < dim; i++)
    free(series[i]);
  free(series);
  free(min);
  free(interval);
  for (i = 0; i < cast_size; i++)
    free(cast[i]);
  free(cast);
  free(newcast);
  free(ctx.list);
  free(ctx.found);
  for (i = 0; i < NMAX; i++)
    free(ctx.box[i]);
  free(ctx.box);
  for (i = 0; i < (unsigned long)dim * embed; i++)
    free(ctx.mat[i]);
  free(ctx.mat);
  free(ctx.vec);
  free(ctx.localav);
  free(ctx.foreav);

  check_alloc(result = (LfoRun *)malloc(sizeof(LfoRun)));
  result->dim = dim;
  result->length = actual_count;
  check_alloc(result->series = (double *)malloc(sizeof(double) * actual_count * dim));
  for (i = 0; i < actual_count; i++) {
    for (j = 0; j < (long)dim; j++)
      result->series[i * dim + j] = out[i][j];
    free(out[i]);
  }
  free(out);

  if (escaped && error != NULL)
    *error = LFO_RUN_ERR_ESCAPED_REGION;

  return result;
}

void lfo_run_free(LfoRun *result)
{
  if (result == NULL)
    return;
  free(result->series);
  free(result);
}
