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

/* Reentrant core of arima-model, factored out of source_c/arima-model.c so
   it has no dependency on argv parsing, file-scope globals, or the
   process-exiting error paths of the routines it used to call (variance(),
   and rescale_data() via rand_arb_dist()). The math here (build_matrix/
   build_vector/multiply_matrix_vector/make_residuals/make_ar_index/
   make_arima_index/iterate_model/iterate_arima_model) is unchanged from
   main(), just parameterized instead of reading/writing globals. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include "../routines/tsa.h"
#include "../../include/arima_model.h"

static unsigned int *make_ar_index_ids(unsigned int poles, unsigned int dim)
{
  unsigned long i, ardim = (unsigned long)poles * dim;
  unsigned int *id;

  check_alloc(id = (unsigned int *)malloc(sizeof(unsigned int) * ardim));
  for (i = 0; i < ardim; i++)
    id[i] = i / poles;
  return id;
}

static unsigned int *make_ar_index_lags(unsigned int poles, unsigned int dim)
{
  unsigned long i, ardim = (unsigned long)poles * dim;
  unsigned int *lag;

  check_alloc(lag = (unsigned int *)malloc(sizeof(unsigned int) * ardim));
  for (i = 0; i < ardim; i++)
    lag[i] = i % poles;
  return lag;
}

static void make_arima_index(unsigned int arpoles, unsigned int mapoles, unsigned int dim,
			      unsigned int **out_id, unsigned int **out_lag)
{
  unsigned long armad = (unsigned long)(arpoles + mapoles) * dim;
  unsigned long i, i0;
  unsigned int *id, *lag;

  check_alloc(id = (unsigned int *)malloc(sizeof(unsigned int) * armad));
  check_alloc(lag = (unsigned int *)malloc(sizeof(unsigned int) * armad));

  for (i = 0; i < (unsigned long)arpoles * dim; i++) {
    id[i] = i / arpoles;
    lag[i] = i % arpoles;
  }
  i0 = (unsigned long)arpoles * dim;
  for (i = 0; i < (unsigned long)mapoles * dim; i++) {
    id[i + i0] = dim + i / mapoles;
    lag[i + i0] = i % mapoles;
  }

  *out_id = id;
  *out_lag = lag;
}

/* series has size >= (max(aindex_id)+1) rows; for the initial AR fit that's
   `series` itself (dim rows), for the ARMA refinement it's a 2*dim-row
   [series;running-residual] array (see arima_model_fit() below). */
static double **build_matrix(double *const *series, unsigned long length, unsigned int poles,
			      unsigned int offset, const unsigned int *aindex_id,
			      const unsigned int *aindex_lag, unsigned int size)
{
  long n, i, j;
  unsigned int id, is, jd, js;
  double norm;
  double **mat;

  check_alloc(mat = (double **)malloc(sizeof(double *) * size));
  for (i = 0; i < (long)size; i++)
    check_alloc(mat[i] = (double *)malloc(sizeof(double) * size));

  norm = 1. / ((double)length - 1.0 - (double)poles - (double)offset);

  for (i = 0; i < (long)size; i++) {
    id = aindex_id[i];
    is = aindex_lag[i];
    for (j = i; j < (long)size; j++) {
      jd = aindex_id[j];
      js = aindex_lag[j];
      mat[i][j] = 0.0;
      for (n = (long)(offset + poles) - 1; n < (long)length - 1; n++)
	mat[i][j] += series[id][n - is] * series[jd][n - js];
      mat[i][j] *= norm;
      mat[j][i] = mat[i][j];
    }
  }

  return mat;
}

static void build_vector(double *vec, unsigned int size, long comp, double *const *series,
			  unsigned long length, unsigned int poles, unsigned int offset,
			  const unsigned int *aindex_id, const unsigned int *aindex_lag)
{
  long i, n;
  unsigned int id, is;
  double norm;

  norm = 1. / ((double)length - 1.0 - (double)poles - (double)offset);

  for (i = 0; i < (long)size; i++) {
    id = aindex_id[i];
    is = aindex_lag[i];
    vec[i] = 0.0;
    for (n = (long)(offset + poles) - 1; n < (long)length - 1; n++)
      vec[i] += series[comp][n + 1] * series[id][n - is];
    vec[i] *= norm;
  }
}

static double *multiply_matrix_vector(double **mat, double *vec, unsigned int size)
{
  long i, j;
  double *new_vec;

  check_alloc(new_vec = (double *)malloc(sizeof(double) * size));

  for (i = 0; i < (long)size; i++) {
    new_vec[i] = 0.0;
    for (j = 0; j < (long)size; j++)
      new_vec[i] += mat[i][j] * vec[j];
  }

  return new_vec;
}

/* diff is [dim][length]; entries [poles..length-1] are (re)written, entries
   before that are left untouched (the caller calloc's diff once so those
   stay zero across every call, matching what the CLI itself never prints:
   the CLI's own residual output always starts at whatever `poles` ends up
   being once fitting is done). series has size >= (max(aindex_id)+1) rows,
   same as build_matrix() above. */
static double *make_residuals(double **diff, double **coeff, unsigned int size,
			       double *const *series, unsigned long length, unsigned int dim,
			       unsigned int poles, const unsigned int *aindex_id,
			       const unsigned int *aindex_lag)
{
  long n, n1, d, i;
  unsigned int id, is;
  double *resi;

  check_alloc(resi = (double *)malloc(sizeof(double) * dim));
  for (i = 0; i < (long)dim; i++)
    resi[i] = 0.0;

  /* poles == 0 only happens via an ARMA refinement with arpoles == mapoles
     == 0 (a degenerate "differencing only" configuration): the original's
     `for (n=poles-1; ...)` computes poles-1 in unsigned arithmetic there,
     which wraps to a huge value and skips the loop entirely (leaving diff
     untouched) rather than underflowing to -1, so replicate that skip here
     instead of reading series[id][n-is] out of bounds for negative n. */
  if (poles > 0) {
    for (n = (long)poles - 1; n < (long)length - 1; n++) {
      n1 = n + 1;
      for (d = 0; d < (long)dim; d++) {
	diff[d][n1] = series[d][n1];
	for (i = 0; i < (long)size; i++) {
	  id = aindex_id[i];
	  is = aindex_lag[i];
	  diff[d][n1] -= coeff[d][i] * series[id][n - is];
	}
	resi[d] += sqr(diff[d][n1]);
      }
    }
  }

  for (i = 0; i < (long)dim; i++)
    resi[i] = sqrt(resi[i] / ((double)length - (double)poles));

  return resi;
}

/* Reentrant reimplementation of rand_arb_dist() (routines/rand_arb_dist.c):
   creates nc random numbers with the empirical distribution of x[0..nx-1].
   Unlike the original, this neither mutates x in place (it works off of a
   private rescaled copy) nor calls rescale_data() - which would exit() the
   process if x were constant - instead returning NULL in that case. */
static double *rand_arb_dist_reentrant(const double *x, unsigned long nx, unsigned long nc,
					unsigned int nb, unsigned long iseed, double *bad_value)
{
  double h, min, inter, *randarb, drnd, epsinv = 1.0 / (double)nb;
  unsigned long i, j, hrnd, nall = nx + nb, *box;
  double *rescaled;

  min = inter = x[0];
  for (i = 1; i < nx; i++) {
    if (x[i] < min)
      min = x[i];
    if (x[i] > inter)
      inter = x[i];
  }
  inter -= min;

  if (inter == 0.0) {
    if (bad_value != NULL)
      *bad_value = min;
    return NULL;
  }

  check_alloc(rescaled = (double *)malloc(sizeof(double) * nx));
  for (i = 0; i < nx; i++)
    rescaled[i] = (x[i] - min) / inter;

  check_alloc(box = (unsigned long *)malloc(sizeof(unsigned long) * nb));
  for (i = 0; i < nb; i++)
    box[i] = 1;

  for (i = 0; i < nx; i++) {
    h = rescaled[i];
    if (h >= 1.0)
      h -= epsinv / 2.0;
    j = (unsigned int)(h * nb);
    box[j]++;
  }
  for (i = 1; i < nb; i++)
    box[i] += box[i - 1];

  check_alloc(randarb = (double *)malloc(sizeof(double) * nc));

  rnd_init(iseed);
  for (i = 0; i < 1000; i++)
    rnd_long();

  for (i = 0; i < nc; i++) {
    hrnd = rnd_long() % nall;
    for (j = 0; j < nb; j++)
      if (box[j] >= hrnd)
	break;
    drnd = (double)rnd_long() / (double)ULONG_MAX * epsinv;
    randarb[i] = min + ((double)j * epsinv + drnd) * inter;
  }

  free(box);
  free(rescaled);
  return randarb;
}

ArimaModel *arima_model_fit(double *const *series, unsigned long length,
			     unsigned int dim, unsigned int poles,
			     int run_arima, unsigned int arpoles,
			     unsigned int mapoles, unsigned int max_iter,
			     double convergence)
{
  long i, j;
  unsigned int *aindex_id, *aindex_lag;
  unsigned int ardim = poles * dim;
  double *vec, **mat, **inverse, **coeff, **diff, *pm;
  unsigned int size, iter_poles, realiter = 0;
  double **xdiff = NULL, *diffcoeff = NULL;
  ArimaModel *model;

  if (poles < 1 || poles >= length)
    return NULL;
  if (run_arima && (arpoles >= length || mapoles >= length))
    return NULL;

  aindex_id = make_ar_index_ids(poles, dim);
  aindex_lag = make_ar_index_lags(poles, dim);

  check_alloc(vec = (double *)malloc(sizeof(double) * ardim));
  mat = build_matrix(series, length, poles, 0, aindex_id, aindex_lag, ardim);
  inverse = invert_matrix(mat, ardim);

  check_alloc(coeff = (double **)malloc(sizeof(double *) * dim));
  for (i = 0; i < dim; i++) {
    build_vector(vec, ardim, i, series, length, poles, 0, aindex_id, aindex_lag);
    coeff[i] = multiply_matrix_vector(inverse, vec, ardim);
  }

  check_alloc(diff = (double **)malloc(sizeof(double *) * dim));
  for (i = 0; i < dim; i++)
    check_alloc(diff[i] = (double *)calloc(length, sizeof(double)));

  pm = make_residuals(diff, coeff, ardim, series, length, dim, poles, aindex_id, aindex_lag);

  free(vec);
  for (i = 0; i < ardim; i++) {
    free(mat[i]);
    free(inverse[i]);
  }
  free(mat);
  free(inverse);

  size = ardim;
  iter_poles = poles;

  if (run_arima) {
    unsigned int offset = poles;
    unsigned int armadim = (arpoles + mapoles) * dim;
    unsigned int poles_eff = (arpoles > mapoles) ? arpoles : mapoles;
    unsigned int iter;
    double **workseries, **oldcoeff, **full_xdiff, *full_diffcoeff;
    double hdiff, alldiff;

    free(aindex_id);
    free(aindex_lag);
    make_arima_index(arpoles, mapoles, dim, &aindex_id, &aindex_lag);

    check_alloc(workseries = (double **)malloc(sizeof(double *) * 2 * dim));
    for (i = 0; i < dim; i++) {
      check_alloc(workseries[i] = (double *)malloc(sizeof(double) * length));
      check_alloc(workseries[i + dim] = (double *)malloc(sizeof(double) * length));
      for (j = 0; j < (long)length; j++) {
	workseries[i][j] = series[i][j];
	workseries[i + dim][j] = diff[i][j];
      }
    }

    check_alloc(oldcoeff = (double **)malloc(sizeof(double *) * dim));
    for (i = 0; i < dim; i++) {
      check_alloc(oldcoeff[i] = (double *)malloc(sizeof(double) * armadim));
      for (j = 0; j < armadim; j++)
	oldcoeff[i][j] = 0.0;
    }

    check_alloc(full_xdiff = (double **)malloc(sizeof(double *) * (max_iter ? max_iter : 1)));
    for (i = 0; i < (long)max_iter; i++)
      check_alloc(full_xdiff[i] = (double *)malloc(sizeof(double) * dim));
    check_alloc(full_diffcoeff = (double *)malloc(sizeof(double) * (max_iter ? max_iter : 1)));

    for (iter = 1; iter <= max_iter; iter++) {
      offset += poles_eff;

      check_alloc(vec = (double *)malloc(sizeof(double) * armadim));
      mat = build_matrix((double *const *)workseries, length, poles_eff, offset,
			  aindex_id, aindex_lag, armadim);
      inverse = invert_matrix(mat, armadim);

      /* This iteration's coeff replaces the previous one; the previous
	 array (the initial AR fit's, on iteration 1, or the prior
	 iteration's otherwise) is freed once we're done reading it below. */
      {
	double **new_coeff;

	check_alloc(new_coeff = (double **)malloc(sizeof(double *) * dim));
	for (i = 0; i < dim; i++) {
	  build_vector(vec, armadim, i, (double *const *)workseries, length, poles_eff,
		       offset, aindex_id, aindex_lag);
	  new_coeff[i] = multiply_matrix_vector(inverse, vec, armadim);
	}

	free(pm);
	pm = make_residuals(diff, new_coeff, armadim, (double *const *)workseries, length,
			     dim, poles_eff, aindex_id, aindex_lag);

	for (j = 0; j < dim; j++) {
	  long hj = j + dim;

	  hdiff = 0.0;
	  for (i = offset; i < (long)length; i++)
	    hdiff += sqr(workseries[hj][i] - diff[j][i]);
	  for (i = 0; i < (long)length; i++)
	    workseries[hj][i] = diff[j][i];
	  full_xdiff[iter - 1][j] = sqrt(hdiff / (double)(length - offset));
	}

	full_diffcoeff[iter - 1] = 0.0;
	for (i = 0; i < dim; i++)
	  for (j = 0; j < dim; j++) {
	    full_diffcoeff[iter - 1] += sqr(new_coeff[i][j] - oldcoeff[i][j]);
	    oldcoeff[i][j] = new_coeff[i][j];
	  }
	full_diffcoeff[iter - 1] = sqrt(full_diffcoeff[iter - 1] / (double)armadim);

	for (i = 0; i < dim; i++)
	  free(coeff[i]);
	free(coeff);
	coeff = new_coeff;
      }

      free(vec);
      for (i = 0; i < armadim; i++) {
	free(mat[i]);
	free(inverse[i]);
      }
      free(mat);
      free(inverse);

      alldiff = full_xdiff[iter - 1][0];
      for (i = 1; i < dim; i++)
	if (full_xdiff[iter - 1][i] > alldiff)
	  alldiff = full_xdiff[iter - 1][i];
      realiter = iter;
      if (alldiff < convergence)
	break;
    }

    for (i = 0; i < dim; i++) {
      free(workseries[i]);
      free(workseries[i + dim]);
    }
    free(workseries);
    for (i = 0; i < dim; i++)
      free(oldcoeff[i]);
    free(oldcoeff);

    check_alloc(xdiff = (double **)malloc(sizeof(double *) * (realiter ? realiter : 1)));
    for (i = 0; i < (long)realiter; i++) {
      check_alloc(xdiff[i] = (double *)malloc(sizeof(double) * dim));
      memcpy(xdiff[i], full_xdiff[i], sizeof(double) * dim);
    }
    check_alloc(diffcoeff = (double *)malloc(sizeof(double) * (realiter ? realiter : 1)));
    memcpy(diffcoeff, full_diffcoeff, sizeof(double) * realiter);

    for (i = 0; i < (long)max_iter; i++)
      free(full_xdiff[i]);
    free(full_xdiff);
    free(full_diffcoeff);

    size = armadim;
    iter_poles = poles_eff;
  }

  check_alloc(model = (ArimaModel *)malloc(sizeof(ArimaModel)));
  model->dim = dim;
  model->length = length;
  model->poles = poles;
  model->arpoles = arpoles;
  model->mapoles = mapoles;
  model->is_arima = run_arima ? 1 : 0;
  model->iter_poles = iter_poles;
  model->size = size;
  model->aindex_id = aindex_id;
  model->aindex_lag = aindex_lag;
  model->coeff = coeff;
  model->rms_error = pm;
  model->residuals = diff;
  model->realiter = realiter;
  model->xdiff = xdiff;
  model->diffcoeff = diffcoeff;

  return model;
}

void arima_model_free(ArimaModel *model)
{
  unsigned int i;

  if (model == NULL)
    return;
  for (i = 0; i < model->dim; i++) {
    free(model->coeff[i]);
    free(model->residuals[i]);
  }
  free(model->coeff);
  free(model->residuals);
  free(model->rms_error);
  free(model->aindex_id);
  free(model->aindex_lag);
  for (i = 0; i < model->realiter; i++)
    free(model->xdiff[i]);
  free(model->xdiff);
  free(model->diffcoeff);
  free(model);
}

double **arima_model_iterate(const ArimaModel *model, unsigned long ilength,
			      unsigned long seed, double *bad_value)
{
  unsigned int dim = model->dim, poles = model->iter_poles;
  unsigned long i, n;
  long j, i1, i2;
  double **myrand, **iterate, *swap, **out;

  check_alloc(myrand = (double **)malloc(sizeof(double *) * dim));
  for (i = 0; i < dim; i++)
    myrand[i] = NULL;

  for (i = 0; i < dim; i++) {
    myrand[i] = rand_arb_dist_reentrant(model->residuals[i], model->length, ilength + poles,
					 100, seed, bad_value);
    if (myrand[i] == NULL) {
      for (n = 0; n <= i; n++)
	free(myrand[n]);
      free(myrand);
      return NULL;
    }
  }

  rnd_init(seed);
  for (i = 0; i < 1000; i++)
    rnd_long();

  check_alloc(out = (double **)malloc(sizeof(double *) * ilength));

  if (!model->is_arima) {
    check_alloc(iterate = (double **)malloc(sizeof(double *) * (poles + 1)));
    for (i = 0; i <= poles; i++)
      check_alloc(iterate[i] = (double *)malloc(sizeof(double) * dim));

    for (i = 0; i < dim; i++)
      for (j = 0; j < (long)poles; j++)
	iterate[j][i] = myrand[i][j];

    for (n = 0; n < ilength; n++) {
      long d;

      for (d = 0; d < dim; d++) {
	iterate[poles][d] = myrand[d][n + poles];
	for (i1 = 0; i1 < dim; i1++)
	  for (i2 = 0; i2 < (long)poles; i2++)
	    iterate[poles][d] += model->coeff[d][i1 * poles + i2] * iterate[poles - 1 - i2][i1];
      }
      check_alloc(out[n] = (double *)malloc(sizeof(double) * dim));
      for (d = 0; d < dim; d++)
	out[n][d] = iterate[poles][d];

      swap = iterate[0];
      for (i = 0; i < poles; i++)
	iterate[i] = iterate[i + 1];
      iterate[poles] = swap;
    }

    for (i = 0; i <= poles; i++)
      free(iterate[i]);
    free(iterate);
  }
  else {
    check_alloc(iterate = (double **)malloc(sizeof(double *) * (poles + 1)));
    for (i = 0; i <= poles; i++)
      check_alloc(iterate[i] = (double *)malloc(sizeof(double) * 2 * dim));

    for (i = 0; i < dim; i++)
      for (j = 0; j < (long)poles; j++)
	iterate[j][i] = iterate[j][dim + i] = myrand[i][j];

    for (n = 0; n < ilength; n++) {
      unsigned long ii;
      long jj;

      for (ii = 0; ii < dim; ii++)
	iterate[poles][ii] = iterate[poles][ii + dim] = myrand[ii][n + poles];

      for (jj = 0; jj < dim; jj++) {
	long k;

	for (k = 0; k < (long)model->size; k++) {
	  unsigned int id = model->aindex_id[k], is = model->aindex_lag[k];

	  iterate[poles][jj] += model->coeff[jj][k] * iterate[poles - 1 - is][id];
	}
      }

      check_alloc(out[n] = (double *)malloc(sizeof(double) * dim));
      for (ii = 0; ii < dim; ii++)
	out[n][ii] = iterate[poles][ii];

      swap = iterate[0];
      for (i = 0; i < poles; i++)
	iterate[i] = iterate[i + 1];
      iterate[poles] = swap;
    }

    for (i = 0; i <= poles; i++)
      free(iterate[i]);
    free(iterate);
  }

  for (i = 0; i < dim; i++)
    free(myrand[i]);
  free(myrand);

  return out;
}

void arima_model_iterate_free(double **iterated, unsigned long ilength)
{
  unsigned long n;

  if (iterated == NULL)
    return;
  for (n = 0; n < ilength; n++)
    free(iterated[n]);
  free(iterated);
}
