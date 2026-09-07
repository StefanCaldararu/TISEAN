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

/* Reentrant core of ghkss, factored out of source_c/ghkss.c so it has no
   dependency on argv parsing, file-scope globals, or the process-exiting
   error paths in the generic rescale_data()/eigen() library routines it
   used to call. The math (sort()/mmb()/fmn()/make_correction()/
   handle_trend()/set_correction() and the main() loop driving them) is
   unchanged; global state is threaded through a GhkssCtx passed to each
   helper instead. */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../routines/tsa.h"
#include "../../include/ghkss.h"

#define BOX (unsigned int)1024

/* tred2/tql2 do the actual eigendecomposition work and are defined
   (non-static) in routines/eigen.c, but only eigen() - the wrapper around
   them that fprintf+exit()s on non-convergence - is declared in tsa.h.
   Calling tred2/tql2 directly lets make_correction() see tql2's
   convergence flag (ierr) itself and report failure instead of going
   through eigen()'s exit() path (see also source_c/api/pca_api.c, which
   extracted the same pattern first). */
extern int tred2(int *nm, int *n, double *a, double *d, double *e, double *z);
extern int tql2(int *nm, int *n, double *d, double *e, double *z, int *ierr);

typedef struct {
  unsigned long length;
  unsigned int comp, embed, delay, qdim, dim;
  int emb_offset;
  unsigned int ibox;

  double **series; /* [comp][length], rescaled to [0,1); corrected in place
		       as iterations progress */
  double **delta;  /* [comp][length] */
  double **corr;   /* [length][dim] */
  double *metric;  /* [dim] */
  double trace;

  long **box;           /* [BOX][BOX] */
  long *list;            /* [length] */
  unsigned long *flist;   /* [length] */

  unsigned int *index_comp, *index_embed; /* [dim] */

  /* scratch, reused across calls like the original's file-scope globals */
  int *sorted;                  /* [dim] */
  double *av, *eig, *matarray;  /* [dim], [dim], [dim*dim] */
  double **mat;                 /* [dim][dim], view into matarray */
} GhkssCtx;

static void ctx_sort(GhkssCtx *ctx, double *x, int *n)
{
  long i, j, iswap;
  double dswap;
  unsigned int dim = ctx->dim;

  for (i = 0; i < (long)dim; i++)
    n[i] = i;

  for (i = 0; i < (long)dim - 1; i++)
    for (j = i + 1; j < (long)dim; j++)
      if (x[j] > x[i]) {
	dswap = x[i];
	x[i] = x[j];
	x[j] = dswap;
	iswap = n[i];
	n[i] = n[j];
	n[j] = iswap;
      }
}

static void ctx_mmb(GhkssCtx *ctx, double eps)
{
  long i, x, y;
  double ieps = 1.0 / eps;
  unsigned int comp = ctx->comp;
  int emb_offset = ctx->emb_offset;
  unsigned int ibox = ctx->ibox;
  double **series = ctx->series;
  long **box = ctx->box;
  long *list = ctx->list;

  for (x = 0; x < BOX; x++)
    for (y = 0; y < BOX; y++)
      box[x][y] = -1;

  for (i = emb_offset; i < (long)ctx->length; i++) {
    x = (int)(series[0][i] * ieps) & ibox;
    y = (int)(series[comp - 1][i - emb_offset] * ieps) & ibox;
    list[i] = box[x][y];
    box[x][y] = i;
  }
}

static unsigned long ctx_fmn(GhkssCtx *ctx, long which, double eps)
{
  unsigned long nf = 0;
  long i, i1, i2, j, j1, k, k1, li;
  long element;
  double dx = 0.0;
  unsigned int comp = ctx->comp, embed = ctx->embed, delay = ctx->delay;
  int emb_offset = ctx->emb_offset;
  unsigned int ibox = ctx->ibox;
  double **series = ctx->series;
  long **box = ctx->box;
  long *list = ctx->list;
  unsigned long *flist = ctx->flist;

  i = (int)(series[0][which] / eps) & ibox;
  j = (int)(series[comp - 1][which - emb_offset] / eps) & ibox;

  for (i1 = i - 1; i1 <= i + 1; i1++) {
    i2 = i1 & ibox;
    for (j1 = j - 1; j1 <= j + 1; j1++) {
      element = box[i2][j1 & ibox];
      while (element != -1) {
	for (k = 0; k < (long)embed; k++) {
	  k1 = -k * (int)delay;
	  for (li = 0; li < (long)comp; li++) {
	    dx = fabs(series[li][which + k1] - series[li][element + k1]);
	    if (dx > eps)
	      break;
	  }
	  if (dx > eps)
	    break;
	}
	if (dx <= eps)
	  flist[nf++] = element;
	element = list[element];
      }
    }
  }
  return nf;
}

/* Returns 0 if the eigenvalue solver failed to converge, 1 on success. */
static int ctx_make_correction(GhkssCtx *ctx, unsigned long n, unsigned long nf)
{
  long i, i1, i2, j, j1, j2, k, k1, k2, hs;
  double help;
  double *trans, *off;
  int ierr, nm = (int)ctx->dim;
  unsigned int dim = ctx->dim, qdim = ctx->qdim;
  double **series = ctx->series, **mat = ctx->mat, *av = ctx->av;
  double *metric = ctx->metric, *eig = ctx->eig;
  int *sorted = ctx->sorted;
  unsigned int *index_comp = ctx->index_comp, *index_embed = ctx->index_embed;
  unsigned long *flist = ctx->flist;
  double **corr = ctx->corr;

  for (i = 0; i < (long)dim; i++) {
    i1 = index_comp[i];
    i2 = index_embed[i];
    help = 0.0;
    for (j = 0; j < (long)nf; j++)
      help += series[i1][flist[j] - i2];
    av[i] = help / nf;
  }

  for (i = 0; i < (long)dim; i++) {
    i1 = index_comp[i];
    i2 = index_embed[i];
    for (j = i; j < (long)dim; j++) {
      help = 0.0;
      j1 = index_comp[j];
      j2 = index_embed[j];
      for (k = 0; k < (long)nf; k++) {
	hs = flist[k];
	help += series[i1][hs - i2] * series[j1][hs - j2];
      }
      mat[i][j] = (help / nf - av[i] * av[j]) * metric[i] * metric[j];
      mat[j][i] = mat[i][j];
    }
  }

  check_alloc(trans = (double *)malloc(sizeof(double) * nm * nm));
  check_alloc(off = (double *)malloc(sizeof(double) * nm));
  tred2(&nm, &nm, &mat[0][0], eig, off, trans);
  tql2(&nm, &nm, eig, off, trans, &ierr);
  if (ierr != 0) {
    free(trans);
    free(off);
    return 0;
  }
  for (i = 0; i < nm; i++)
    for (j = 0; j < nm; j++)
      mat[i][j] = trans[i + nm * j];
  free(trans);
  free(off);

  ctx_sort(ctx, eig, sorted);

  for (i = 0; i < (long)dim; i++) {
    help = 0.0;
    for (j = qdim; j < (long)dim; j++) {
      hs = sorted[j];
      for (k = 0; k < (long)dim; k++) {
	k1 = index_comp[k];
	k2 = index_embed[k];
	help += (series[k1][n - k2] - av[k]) * mat[k][hs] * mat[i][hs] * metric[k];
      }
    }
    corr[n][i] = help / metric[i];
  }
  return 1;
}

static void ctx_handle_trend(GhkssCtx *ctx, unsigned long n, unsigned long nf)
{
  long i, i1, i2, j;
  double help;
  unsigned int dim = ctx->dim;
  double **corr = ctx->corr, *av = ctx->av;
  unsigned long *flist = ctx->flist;
  unsigned int *index_comp = ctx->index_comp, *index_embed = ctx->index_embed;
  double **delta = ctx->delta, *metric = ctx->metric, trace = ctx->trace;

  for (i = 0; i < (long)dim; i++) {
    help = 0.0;
    for (j = 0; j < (long)nf; j++)
      help += corr[flist[j]][i];
    av[i] = help / nf;
  }

  for (i = 0; i < (long)dim; i++) {
    i1 = index_comp[i];
    i2 = index_embed[i];
    delta[i1][n - i2] += (corr[n][i] - av[i]) / (trace * metric[i]);
  }
}

/* Computes the average shift/rms correction per component (in the
   internally-rescaled [0,1) space, matching set_correction()'s hav/hsigma)
   and subtracts delta from series. Does not touch mineps/resize_eps - the
   caller (ghkss_reduce) owns that, since it also decides whether to grow
   the per-iteration diagnostic arrays. */
static void ctx_set_correction(GhkssCtx *ctx, double *shift, double *rms)
{
  long i, j;
  double *hav, *hsigma, help;
  unsigned int comp = ctx->comp;
  unsigned long length = ctx->length;
  double **delta = ctx->delta, **series = ctx->series;

  check_alloc(hav = (double *)malloc(sizeof(double) * comp));
  check_alloc(hsigma = (double *)malloc(sizeof(double) * comp));
  for (j = 0; j < (long)comp; j++)
    hav[j] = hsigma[j] = 0.0;

  for (i = 0; i < (long)length; i++)
    for (j = 0; j < (long)comp; j++) {
      hav[j] += (help = delta[j][i]);
      hsigma[j] += help * help;
    }

  for (j = 0; j < (long)comp; j++) {
    hav[j] /= length;
    hsigma[j] = sqrt(fabs(hsigma[j] / length - hav[j] * hav[j]));
  }

  for (i = 0; i < (long)comp; i++) {
    shift[i] = hav[i];
    rms[i] = hsigma[i];
  }

  for (i = 0; i < (long)length; i++)
    for (j = 0; j < (long)comp; j++)
      series[j][i] -= delta[j][i];

  free(hav);
  free(hsigma);
}

#define GHKSS_STEP_CHUNK 64

static GHKSSEpsStep *grow_steps(GHKSSEpsStep *steps, unsigned long *capacity)
{
  *capacity += GHKSS_STEP_CHUNK;
  check_alloc(steps = (GHKSSEpsStep *)realloc(steps, sizeof(GHKSSEpsStep) * (*capacity)));
  return steps;
}

static void free_iteration(GHKSSIteration *it, unsigned int comp)
{
  unsigned int j;

  if (it->series != NULL) {
    for (j = 0; j < comp; j++)
      free(it->series[j]);
    free(it->series);
  }
  free(it->shift);
  free(it->rms);
  free(it->correction_steps);
  free(it->trend_steps);
}

static void free_ctx(GhkssCtx *ctx, unsigned int comp, unsigned long length)
{
  unsigned long i;

  for (i = 0; i < comp; i++)
    free(ctx->series[i]);
  free(ctx->series);
  for (i = 0; i < BOX; i++)
    free(ctx->box[i]);
  free(ctx->box);
  free(ctx->list);
  free(ctx->flist);
  free(ctx->metric);
  for (i = 0; i < length; i++)
    free(ctx->corr[i]);
  free(ctx->corr);
  for (i = 0; i < comp; i++)
    free(ctx->delta[i]);
  free(ctx->delta);
  free(ctx->index_comp);
  free(ctx->index_embed);
  free(ctx->av);
  free(ctx->sorted);
  free(ctx->eig);
  free(ctx->matarray);
  free(ctx->mat);
}

GHKSSResult *ghkss_reduce(double *const *series_in, unsigned long length, unsigned int comp,
			   unsigned int embed, unsigned int delay, unsigned int qdim,
			   unsigned int minn, double mineps_in, char eps_set,
			   unsigned int iterations, char euclidean,
			   GHKSSError *error, double *bad_value)
{
  unsigned long i, j;
  unsigned int c;
  long n;
  double min, interval, d_max_max;
  double *d_min, *d_max;
  double mineps, epsfac;
  char resize_eps;
  GhkssCtx ctx;
  GHKSSResult *result;
  int *ok;
  unsigned int iter;

  if (length < minn) {
    if (error != NULL)
      *error = GHKSS_ERR_TOO_MANY_NEIGHBORS;
    return NULL;
  }

  check_alloc(d_min = (double *)malloc(sizeof(double) * comp));
  check_alloc(d_max = (double *)malloc(sizeof(double) * comp));
  check_alloc(ctx.series = (double **)malloc(sizeof(double *) * comp));

  d_max_max = 0.0;
  for (c = 0; c < comp; c++) {
    check_alloc(ctx.series[c] = (double *)malloc(sizeof(double) * length));

    min = interval = series_in[c][0];
    for (i = 1; i < length; i++) {
      if (series_in[c][i] < min)
	min = series_in[c][i];
      if (series_in[c][i] > interval)
	interval = series_in[c][i];
    }
    interval -= min;

    if (interval == 0.0) {
      if (error != NULL)
	*error = GHKSS_ERR_ZERO_INTERVAL;
      if (bad_value != NULL)
	*bad_value = min;
      for (i = 0; i <= c; i++)
	free(ctx.series[i]);
      free(ctx.series);
      free(d_min);
      free(d_max);
      return NULL;
    }

    for (i = 0; i < length; i++)
      ctx.series[c][i] = (series_in[c][i] - min) / interval;
    d_min[c] = min;
    d_max[c] = interval;
    if (d_max[c] > d_max_max)
      d_max_max = d_max[c];
  }

  mineps = eps_set ? mineps_in / d_max_max : 1. / 1000.;
  epsfac = sqrt(2.0);

  ctx.length = length;
  ctx.comp = comp;
  ctx.embed = embed;
  ctx.delay = delay;
  ctx.qdim = qdim;
  ctx.dim = comp * embed;
  ctx.emb_offset = (embed - 1) * delay;
  ctx.ibox = BOX - 1;

  check_alloc(ctx.box = (long **)malloc(sizeof(long *) * BOX));
  for (i = 0; i < BOX; i++)
    check_alloc(ctx.box[i] = (long *)malloc(sizeof(long) * BOX));

  check_alloc(ctx.list = (long *)malloc(sizeof(long) * length));
  check_alloc(ctx.flist = (unsigned long *)malloc(sizeof(long) * length));

  check_alloc(ctx.metric = (double *)malloc(sizeof(double) * ctx.dim));
  ctx.trace = 0.0;
  if (euclidean) {
    for (i = 0; i < ctx.dim; i++) {
      ctx.metric[i] = 1.0;
      ctx.trace += 1. / ctx.metric[i];
    }
  }
  else {
    for (i = 0; i < ctx.dim; i++) {
      if ((i >= comp) && (i < ((long)ctx.dim - (long)comp)))
	ctx.metric[i] = 1.0;
      else
	ctx.metric[i] = 1.0e3;
      ctx.trace += 1. / ctx.metric[i];
    }
  }

  check_alloc(ctx.corr = (double **)malloc(sizeof(double *) * length));
  for (i = 0; i < length; i++)
    check_alloc(ctx.corr[i] = (double *)malloc(sizeof(double) * ctx.dim));
  check_alloc(ok = (int *)malloc(sizeof(int) * length));
  check_alloc(ctx.delta = (double **)malloc(sizeof(double *) * comp));
  for (i = 0; i < comp; i++)
    check_alloc(ctx.delta[i] = (double *)malloc(sizeof(double) * length));
  check_alloc(ctx.index_comp = (unsigned int *)malloc(sizeof(int) * ctx.dim));
  check_alloc(ctx.index_embed = (unsigned int *)malloc(sizeof(int) * ctx.dim));
  check_alloc(ctx.av = (double *)malloc(sizeof(double) * ctx.dim));
  check_alloc(ctx.sorted = (int *)malloc(sizeof(int) * ctx.dim));
  check_alloc(ctx.eig = (double *)malloc(sizeof(double) * ctx.dim));
  check_alloc(ctx.matarray = (double *)malloc(sizeof(double) * ctx.dim * ctx.dim));
  check_alloc(ctx.mat = (double **)malloc(sizeof(double *) * ctx.dim));
  for (i = 0; i < ctx.dim; i++)
    ctx.mat[i] = (double *)(ctx.matarray + ctx.dim * i);

  for (i = 0; i < ctx.dim; i++) {
    ctx.index_comp[i] = i % comp;
    ctx.index_embed[i] = (i / comp) * delay;
  }

  check_alloc(result = (GHKSSResult *)malloc(sizeof(GHKSSResult)));
  result->comp = comp;
  result->length = length;
  result->iterations = iterations;
  check_alloc(result->iters = (GHKSSIteration *)malloc(sizeof(GHKSSIteration) * iterations));

  resize_eps = 0;

  for (iter = 0; iter < iterations; iter++) {
    unsigned int epscount;
    unsigned long allfound, nfound;
    double epsilon;
    char all_done;
    GHKSSEpsStep *csteps, *tsteps;
    unsigned long n_csteps, csteps_cap;
    double *shift, *rms;
    double **out_series;
    char eigen_failed;

    for (i = 0; i < length; i++) {
      ok[i] = 0;
      for (j = 0; j < ctx.dim; j++)
	ctx.corr[i][j] = 0.0;
      for (j = 0; j < comp; j++)
	ctx.delta[j][i] = 0.0;
    }

    epsilon = mineps;
    all_done = 0;
    epscount = 1;
    allfound = 0;
    eigen_failed = 0;

    n_csteps = 0;
    csteps_cap = 0;
    csteps = NULL;

    while (!all_done) {
      ctx_mmb(&ctx, epsilon);
      all_done = 1;
      for (n = ctx.emb_offset; n < (long)length; n++)
	if (!ok[n]) {
	  nfound = ctx_fmn(&ctx, n, epsilon);
	  if (nfound >= minn) {
	    if (!ctx_make_correction(&ctx, n, nfound)) {
	      eigen_failed = 1;
	      break;
	    }
	    ok[n] = epscount;
	    if (epscount == 1)
	      resize_eps = 1;
	    allfound++;
	  }
	  else
	    all_done = 0;
	}
      if (eigen_failed)
	break;

      if (n_csteps == csteps_cap)
	csteps = grow_steps(csteps, &csteps_cap);
      csteps[n_csteps].epsilon = epsilon * d_max_max;
      csteps[n_csteps].count = allfound;
      n_csteps++;

      epsilon *= epsfac;
      epscount++;
    }

    if (eigen_failed) {
      free(csteps);
      for (i = 0; i < iter; i++)
	free_iteration(&result->iters[i], comp);
      free(result->iters);
      free(result);
      free(ok);
      goto free_ctx_and_fail;
    }

    check_alloc(tsteps = (GHKSSEpsStep *)malloc(sizeof(GHKSSEpsStep) * n_csteps));
    epsilon = mineps;
    allfound = 0;
    for (i = 1; i < epscount; i++) {
      ctx_mmb(&ctx, epsilon);
      for (n = ctx.emb_offset; n < (long)length; n++)
	if (ok[n] == (int)i) {
	  nfound = ctx_fmn(&ctx, n, epsilon);
	  ctx_handle_trend(&ctx, n, nfound);
	  allfound++;
	}
      tsteps[i - 1].epsilon = epsilon * d_max_max;
      tsteps[i - 1].count = allfound;
      epsilon *= epsfac;
    }

    check_alloc(shift = (double *)malloc(sizeof(double) * comp));
    check_alloc(rms = (double *)malloc(sizeof(double) * comp));
    ctx_set_correction(&ctx, shift, rms);
    for (c = 0; c < comp; c++) {
      shift[c] *= d_max[c];
      rms[c] *= d_max[c];
    }

    result->iters[iter].mineps_reset = resize_eps ? 1 : 0;
    if (resize_eps)
      mineps /= epsfac;
    resize_eps = 0;
    result->iters[iter].mineps_after = mineps * d_max_max;

    check_alloc(out_series = (double **)malloc(sizeof(double *) * comp));
    for (c = 0; c < comp; c++) {
      check_alloc(out_series[c] = (double *)malloc(sizeof(double) * length));
      for (i = 0; i < length; i++)
	out_series[c][i] = ctx.series[c][i] * d_max[c] + d_min[c];
    }

    result->iters[iter].shift = shift;
    result->iters[iter].rms = rms;
    result->iters[iter].correction_steps = csteps;
    result->iters[iter].n_correction_steps = n_csteps;
    result->iters[iter].trend_steps = tsteps;
    result->iters[iter].n_trend_steps = n_csteps;
    result->iters[iter].series = out_series;
  }

  free(ok);
  free_ctx(&ctx, comp, length);
  free(d_min);
  free(d_max);

  return result;

free_ctx_and_fail:
  free_ctx(&ctx, comp, length);
  free(d_min);
  free(d_max);
  if (error != NULL)
    *error = GHKSS_ERR_EIGEN_NO_CONVERGE;
  return NULL;
}

void ghkss_free(GHKSSResult *result)
{
  unsigned int i;

  if (result == NULL)
    return;
  for (i = 0; i < result->iterations; i++)
    free_iteration(&result->iters[i], result->comp);
  free(result->iters);
  free(result);
}
