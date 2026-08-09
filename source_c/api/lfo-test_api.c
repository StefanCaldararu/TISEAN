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

/* Reentrant core of lfo-test, factored out of source_c/lfo-test.c so it has
   no dependency on argv parsing, file-scope globals, or the process-exiting
   error paths in the generic variance()/rescale_data() library routines it
   used to call. The math here (the per-component rescale, the box-assisted
   neighbor search in put_in_boxes()/hfind_neighbors(), the local-linear fit
   in multiply_matrix()/make_fit()) is unchanged from the original - this
   includes make_fit()'s vec[] accumulation not centering by the neighbor
   average of the second series the way its localav/mat accumulations do;
   that asymmetry is in the original math and is preserved here. */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../routines/tsa.h"
#include "../../include/lfo-test.h"

/* number of boxes for the neighbor search algorithm - must be a power of
   two, since box indices are formed with "& (NMAX - 1)" instead of "% NMAX" */
#define NMAX 512

/* Bundles what put_in_boxes()/hfind_neighbors()/make_fit() used to reach
   through file-scope globals, so those functions can stay reentrant. */
typedef struct {
  double **series;    /* [comp][length], rescaled to [0,1) */
  unsigned int comp, comp1, dim, embed, delay;
  unsigned long length;
  unsigned int minn, step;
  unsigned long causal;
  unsigned long hdim;   /* (embed-1)*delay */
  unsigned int **indexes; /* [2][dim], from make_multi_index() */
  long **box;              /* [NMAX][NMAX] */
  long *list;                /* [length] */
  unsigned long *found, *hfound; /* [length] */
  double epsilon;
  double **mat;                    /* [dim][dim] scratch matrix */
  double *vec, *localav, *foreav;   /* scratch vectors for make_fit() */
} LfoTestCtx;

static void lfo_test_put_in_boxes(LfoTestCtx *ctx)
{
  long i, j, n;
  long hdim = (long)ctx->hdim;
  unsigned long limit = ctx->length - ctx->step;
  double epsinv;

  epsinv = 1.0 / ctx->epsilon;
  for (i = 0; i < NMAX; i++)
    for (j = 0; j < NMAX; j++)
      ctx->box[i][j] = -1;

  for (n = hdim; n < (long)limit; n++) {
    i = (long)(ctx->series[0][n] * epsinv) & (NMAX - 1);
    j = (long)(ctx->series[ctx->comp1][n - hdim] * epsinv) & (NMAX - 1);
    ctx->list[n] = ctx->box[i][j];
    ctx->box[i][j] = n;
  }
}

static unsigned int lfo_test_hfind_neighbors(LfoTestCtx *ctx, unsigned long act)
{
  char toolarge;
  long i, j, i1, i2, j1, k, element;
  long hdim = (long)ctx->hdim;
  unsigned long nfound = 0;
  unsigned int hcomp, hdel;
  double max, dx, epsinv;

  epsinv = 1.0 / ctx->epsilon;

  i = (long)(ctx->series[0][act] * epsinv) & (NMAX - 1);
  j = (long)(ctx->series[ctx->comp1][act - hdim] * epsinv) & (NMAX - 1);

  for (i1 = i - 1; i1 <= i + 1; i1++) {
    i2 = i1 & (NMAX - 1);
    for (j1 = j - 1; j1 <= j + 1; j1++) {
      element = ctx->box[i2][j1 & (NMAX - 1)];
      while (element != -1) {
	max = 0.0;
	toolarge = 0;
	for (k = 0; k < (long)ctx->dim; k++) {
	  hcomp = ctx->indexes[0][k];
	  hdel = ctx->indexes[1][k];
	  dx = fabs(ctx->series[hcomp][element - hdel] - ctx->series[hcomp][act - hdel]);
	  max = (dx > max) ? dx : max;
	  if (max > ctx->epsilon) {
	    toolarge = 1;
	    break;
	  }
	  if (toolarge)
	    break;
	}
	if (max <= ctx->epsilon)
	  ctx->hfound[nfound++] = element;
	element = ctx->list[element];
      }
    }
  }
  return nfound;
}

static void lfo_test_multiply_matrix(LfoTestCtx *ctx, double **mat, double *vec)
{
  double *hvec;
  long i, j;
  long dim = (long)ctx->dim;

  check_alloc(hvec = (double *)malloc(sizeof(double) * dim));
  for (i = 0; i < dim; i++) {
    hvec[i] = 0.0;
    for (j = 0; j < dim; j++)
      hvec[i] += mat[i][j] * vec[j];
  }
  for (i = 0; i < dim; i++)
    vec[i] = hvec[i];
  free(hvec);
}

static void lfo_test_make_fit(LfoTestCtx *ctx, unsigned long number,
			       unsigned long act, double *newcast)
{
  double *sj, *si, lavi, lavj, fav;
  unsigned int hci, hdi, hcj, hdj;
  long i, j, n, which;
  long dim = (long)ctx->dim, comp = (long)ctx->comp;
  double **imat;

  for (i = 0; i < dim; i++)
    ctx->localav[i] = 0.0;
  for (i = 0; i < comp; i++)
    ctx->foreav[i] = 0.0;

  for (n = 0; n < (long)number; n++) {
    which = ctx->found[n];
    for (j = 0; j < comp; j++)
      ctx->foreav[j] += ctx->series[j][which + ctx->step];
    for (j = 0; j < dim; j++) {
      hcj = ctx->indexes[0][j];
      hdj = ctx->indexes[1][j];
      ctx->localav[j] += ctx->series[hcj][which - hdj];
    }
  }

  for (i = 0; i < dim; i++)
    ctx->localav[i] /= number;
  for (i = 0; i < comp; i++)
    ctx->foreav[i] /= number;

  for (i = 0; i < dim; i++) {
    hci = ctx->indexes[0][i];
    hdi = ctx->indexes[1][i];
    lavi = ctx->localav[i];
    si = ctx->series[hci];
    for (j = i; j < dim; j++) {
      hcj = ctx->indexes[0][j];
      hdj = ctx->indexes[1][j];
      lavj = ctx->localav[j];
      sj = ctx->series[hcj];
      ctx->mat[i][j] = 0.0;
      for (n = 0; n < (long)number; n++) {
	which = ctx->found[n];
	ctx->mat[i][j] += (si[which - hdi] - lavi) * (sj[which - hdj] - lavj);
      }
      ctx->mat[i][j] /= number;
      ctx->mat[j][i] = ctx->mat[i][j];
    }
  }

  imat = invert_matrix(ctx->mat, ctx->dim);

  for (i = 0; i < comp; i++) {
    si = ctx->series[i];
    fav = ctx->foreav[i];
    for (j = 0; j < dim; j++) {
      hcj = ctx->indexes[0][j];
      hdj = ctx->indexes[1][j];
      lavj = ctx->localav[j];
      ctx->vec[j] = 0.0;
      sj = ctx->series[hcj];
      for (n = 0; n < (long)number; n++) {
	which = ctx->found[n];
	/* Note: unlike localav/mat above, this does not subtract lavj - that
	   asymmetry is present in the original code and is preserved. */
	ctx->vec[j] += (si[which + ctx->step] - fav) * (sj[which - hdj]);
      }
      ctx->vec[j] /= number;
    }

    lfo_test_multiply_matrix(ctx, imat, ctx->vec);

    newcast[i] = ctx->foreav[i];
    for (j = 0; j < dim; j++) {
      hcj = ctx->indexes[0][j];
      hdj = ctx->indexes[1][j];
      newcast[i] += ctx->vec[j] * (ctx->series[hcj][act - hdj] - ctx->localav[j]);
    }
  }

  for (i = 0; i < dim; i++)
    free(imat[i]);
  free(imat);
}

LfoTest *lfo_test_forecast(double *const *series_in, unsigned long length,
			    unsigned int comp, unsigned int embed,
			    unsigned int delay, unsigned int minn,
			    unsigned int step, unsigned long causal,
			    unsigned long iterations,
			    double eps0, char epsset, double epsf,
			    double *bad_value)
{
  unsigned int c;
  unsigned long i, hdim, clength;
  long li, lj;
  double *interval, *rms;
  double min, max, average, var, h, maxinterval, epsilon, norm;
  double *error, *newcast;
  unsigned long actfound;
  char *done;
  double **series;
  LfoTestCtx ctx;
  LfoTest *result;
  char alldone;

  if (comp == 0 || embed == 0 || length == 0)
    return NULL;

  hdim = (unsigned long)(embed - 1) * delay;
  if (hdim >= length || length - hdim < minn)
    return NULL;

  check_alloc(series = (double **)malloc(sizeof(double *) * comp));
  check_alloc(interval = (double *)malloc(sizeof(double) * comp));
  check_alloc(rms = (double *)malloc(sizeof(double) * comp));

  maxinterval = 0.0;
  for (c = 0; c < comp; c++) {
    check_alloc(series[c] = (double *)malloc(sizeof(double) * length));
    for (i = 0; i < length; i++)
      series[c][i] = series_in[c][i];

    /* rescale_data(series[c], length, &min, &interval[c]), on our own
       private copy */
    min = max = series[c][0];
    for (i = 1; i < length; i++) {
      if (series[c][i] < min) min = series[c][i];
      if (series[c][i] > max) max = series[c][i];
    }
    max -= min;
    if (max == 0.0) {
      unsigned int already_allocated;

      if (bad_value != NULL)
	*bad_value = min;
      for (already_allocated = 0; already_allocated <= c; already_allocated++)
	free(series[already_allocated]);
      free(series);
      free(interval);
      free(rms);
      return NULL;
    }
    for (i = 0; i < length; i++)
      series[c][i] = (series[c][i] - min) / max;
    interval[c] = max;
    if (max > maxinterval)
      maxinterval = max;

    /* variance(series[c], length, &average, &rms[c]), on the now-rescaled
       series. Since interval != 0 above, the rescaled series can't be
       constant, so this can never hit variance()'s own zero-variance exit
       path the way rescale_data()'s can - unlike histogram_compute(), this
       needs no second check. */
    average = var = 0.0;
    for (i = 0; i < length; i++) {
      h = series[c][i];
      average += h;
      var += h * h;
    }
    average /= (double)length;
    rms[c] = sqrt(fabs(var / (double)length - average * average));
  }

  if (epsset)
    eps0 /= maxinterval;

  clength = (iterations <= length) ? iterations - step : length - step;

  check_alloc(ctx.list = (long *)malloc(sizeof(long) * length));
  check_alloc(ctx.found = (unsigned long *)malloc(sizeof(unsigned long) * length));
  check_alloc(ctx.hfound = (unsigned long *)malloc(sizeof(unsigned long) * length));
  check_alloc(done = (char *)malloc(sizeof(char) * length));
  check_alloc(ctx.box = (long **)malloc(sizeof(long *) * NMAX));
  for (li = 0; li < NMAX; li++)
    check_alloc(ctx.box[li] = (long *)malloc(sizeof(long) * NMAX));

  for (i = 0; i < length; i++)
    done[i] = 0;

  ctx.series = series;
  ctx.comp = comp;
  ctx.comp1 = comp - 1;
  ctx.dim = embed * comp;
  ctx.embed = embed;
  ctx.delay = delay;
  ctx.length = length;
  ctx.minn = minn;
  ctx.step = step;
  ctx.causal = causal;
  ctx.hdim = hdim;
  ctx.indexes = make_multi_index(comp, embed, delay);

  check_alloc(newcast = (double *)malloc(sizeof(double) * comp));
  check_alloc(ctx.localav = (double *)malloc(sizeof(double) * ctx.dim));
  check_alloc(ctx.foreav = (double *)malloc(sizeof(double) * comp));
  check_alloc(ctx.vec = (double *)malloc(sizeof(double) * ctx.dim));
  check_alloc(ctx.mat = (double **)malloc(sizeof(double *) * ctx.dim));
  for (li = 0; li < (long)ctx.dim; li++)
    check_alloc(ctx.mat[li] = (double *)malloc(sizeof(double) * ctx.dim));

  check_alloc(error = (double *)malloc(sizeof(double) * comp));
  for (c = 0; c < comp; c++)
    error[c] = 0.0;

  check_alloc(result = (LfoTest *)malloc(sizeof(LfoTest)));
  result->comp = comp;
  result->length = length;
  check_alloc(result->rms_error = (double *)malloc(sizeof(double) * comp));
  check_alloc(result->individual = (double *)malloc(sizeof(double) * comp * length));
  for (i = 0; i < (unsigned long)comp * length; i++)
    result->individual[i] = 0.0;

  epsilon = eps0 / epsf;
  alldone = 0;
  while (!alldone) {
    alldone = 1;
    epsilon *= epsf;
    ctx.epsilon = epsilon;
    lfo_test_put_in_boxes(&ctx);
    for (li = (long)hdim; li < (long)clength; li++) {
      if (!done[li]) {
	actfound = lfo_test_hfind_neighbors(&ctx, (unsigned long)li);
	actfound = exclude_interval(actfound, li - (long)causal + 1,
				     li + (long)causal + (long)hdim - 1,
				     ctx.hfound, ctx.found);
	if (actfound > minn) {
	  lfo_test_make_fit(&ctx, actfound, (unsigned long)li, newcast);
	  for (lj = 0; lj < (long)comp; lj++)
	    error[lj] += sqr(newcast[lj] - series[lj][li + step]);
	  for (lj = 0; lj < (long)comp; lj++)
	    result->individual[lj * length + li] =
	      (newcast[lj] - series[lj][li + step]) * interval[lj];
	  done[li] = 1;
	}
	alldone &= done[li];
      }
    }
  }

  norm = (double)clength - (double)hdim;
  for (c = 0; c < comp; c++)
    result->rms_error[c] = sqrt(error[c] / norm) / rms[c];

  free(newcast);
  free(error);
  free(ctx.localav);
  free(ctx.foreav);
  free(ctx.vec);
  for (li = 0; li < (long)ctx.dim; li++)
    free(ctx.mat[li]);
  free(ctx.mat);
  free(ctx.indexes[0]);
  free(ctx.indexes[1]);
  free(ctx.indexes);
  for (li = 0; li < NMAX; li++)
    free(ctx.box[li]);
  free(ctx.box);
  free(done);
  free(ctx.hfound);
  free(ctx.found);
  free(ctx.list);
  for (c = 0; c < comp; c++)
    free(series[c]);
  free(series);
  free(interval);
  free(rms);

  return result;
}

void lfo_test_free(LfoTest *result)
{
  if (result == NULL)
    return;
  free(result->rms_error);
  free(result->individual);
  free(result);
}
