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
   process-exiting error path in the generic variance() library routine it
   used to call (reimplemented inline here instead - see the centering loop
   in arima_model_fit() below). The math (make_difference()'s differencing,
   build_matrix()/build_vector()/multiply_matrix_vector()/make_residuals()'s
   matrix fit, the ARMA convergence loop, and iterate_model()/
   iterate_arima_model()'s forward iteration) is unchanged from the
   original. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../routines/tsa.h"
#include "../../include/arima-model.h"

static unsigned int **make_ar_index(unsigned int dim, unsigned int poles)
{
  unsigned int **idx;
  unsigned long i, ardim = (unsigned long)poles * dim;

  check_alloc(idx = (unsigned int **)malloc(sizeof(unsigned int *) * 2));
  for (i = 0; i < 2; i++)
    check_alloc(idx[i] = (unsigned int *)malloc(sizeof(unsigned int) * ardim));
  for (i = 0; i < ardim; i++) {
    idx[0][i] = i / poles;
    idx[1][i] = i % poles;
  }
  return idx;
}

static unsigned int **make_arima_index(unsigned int dim, unsigned int ars,
					unsigned int mas)
{
  unsigned int **idx;
  unsigned long armad = (unsigned long)(ars + mas) * dim;
  unsigned long i, i0;

  check_alloc(idx = (unsigned int **)malloc(sizeof(unsigned int *) * 2));
  for (i = 0; i < 2; i++)
    check_alloc(idx[i] = (unsigned int *)malloc(sizeof(unsigned int) * armad));
  for (i = 0; i < (unsigned long)ars * dim; i++) {
    idx[0][i] = i / ars;
    idx[1][i] = i % ars;
  }
  i0 = (unsigned long)ars * dim;
  for (i = 0; i < (unsigned long)mas * dim; i++) {
    idx[0][i + i0] = dim + i / mas;
    idx[1][i + i0] = i % mas;
  }
  return idx;
}

static void free_index(unsigned int **idx)
{
  free(idx[0]);
  free(idx[1]);
  free(idx);
}

/* Fills and inverts the normal-equations matrix for a window of `size`
   (component,lag) terms described by aindex, then frees the (uninverted)
   matrix itself - the original build_matrix() left that to its caller, but
   nothing here ever reads it again once inverse is computed. */
static double **build_matrix(double *const *series, unsigned long length,
			      unsigned int poles, unsigned int offset,
			      unsigned int **aindex, unsigned int size)
{
  long n;
  unsigned int i, j;
  long is, id, js, jd;
  double norm;
  double **mat, **inverse;

  check_alloc(mat = (double **)malloc(sizeof(double *) * size));
  for (i = 0; i < size; i++)
    check_alloc(mat[i] = (double *)malloc(sizeof(double) * size));

  norm = 1. / ((double)length - 1.0 - (double)poles - (double)offset);

  for (i = 0; i < size; i++) {
    id = aindex[0][i];
    is = aindex[1][i];
    for (j = i; j < size; j++) {
      jd = aindex[0][j];
      js = aindex[1][j];
      mat[i][j] = 0.0;
      for (n = (long)offset + (long)poles - 1; n < (long)length - 1; n++)
	mat[i][j] += series[id][n - is] * series[jd][n - js];
      mat[i][j] *= norm;
      mat[j][i] = mat[i][j];
    }
  }

  inverse = invert_matrix(mat, size);

  for (i = 0; i < size; i++)
    free(mat[i]);
  free(mat);

  return inverse;
}

static void build_vector(double *vec, unsigned int size, long comp,
			  double *const *series, unsigned long length,
			  unsigned int poles, unsigned int offset,
			  unsigned int **aindex)
{
  long i, is, id, n;
  double norm;

  norm = 1. / ((double)length - 1.0 - (double)poles - (double)offset);

  for (i = 0; i < (long)size; i++) {
    id = aindex[0][i];
    is = aindex[1][i];
    vec[i] = 0.0;
    for (n = (long)offset + (long)poles - 1; n < (long)length - 1; n++)
      vec[i] += series[comp][n + 1] * series[id][n - is];
    vec[i] *= norm;
  }
}

static double *multiply_matrix_vector(double **mat, double *vec,
				       unsigned int size)
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

/* Fills diff[d][poles..length-1] with the one-step-ahead prediction
   residuals and rms_error[d] with their RMS, for every component d.
   diff[d][0..poles-1] is left untouched by design - the caller calloc's it
   once up front so those entries read as zero rather than whatever a
   previous (different-poles) call happened to leave behind. */
static void make_residuals(double **diff, double **coeff, double *rms_error,
			    double *const *series, unsigned long length,
			    unsigned int dim, unsigned int poles,
			    unsigned int **aindex, unsigned int size)
{
  long n, n1;
  unsigned int d;
  long i, is, id;

  for (d = 0; d < dim; d++)
    rms_error[d] = 0.0;

  for (n = (long)poles - 1; n < (long)length - 1; n++) {
    n1 = n + 1;
    for (d = 0; d < dim; d++) {
      diff[d][n1] = series[d][n1];
      for (i = 0; i < (long)size; i++) {
	id = aindex[0][i];
	is = aindex[1][i];
	diff[d][n1] -= coeff[d][i] * series[id][n - is];
      }
      rms_error[d] += sqr(diff[d][n1]);
    }
  }
  for (d = 0; d < dim; d++)
    rms_error[d] = sqrt(rms_error[d] / ((double)length - (double)poles));
}

ARIMAModel *arima_model_fit(double *const *series_in, unsigned long length_in,
			     unsigned int dim, unsigned int poles,
			     unsigned int arpoles, unsigned int ipoles,
			     unsigned int mapoles, unsigned int iterations,
			     double convergence, ARIMAModelError *error)
{
  unsigned char arimaset = (arpoles + ipoles + mapoles) > 0;
  unsigned long length = length_in;
  unsigned int i, j;
  long n;
  double **series;
  double *average;
  unsigned int ardim, armadim = 0, size, order;
  unsigned int **aindex;
  double *vec, **inverse, **coeff, **diff, *rms_error;
  unsigned int realiter = 0;
  double **xdiff_out = NULL, *diffcoeff_out = NULL;
  ARIMAModel *model;

  if (error != NULL)
    *error = ARIMA_MODEL_OK;

  /* Private working copy: differencing and centering both mutate the
     series in place in the original CLI, and we don't want to mutate the
     caller's array. */
  check_alloc(series = (double **)malloc(sizeof(double *) * dim));
  for (i = 0; i < dim; i++) {
    check_alloc(series[i] = (double *)malloc(sizeof(double) * length));
    memcpy(series[i], series_in[i], sizeof(double) * length);
  }

  /* make_difference(), run ipoles times, in place. */
  for (n = 0; n < (long)ipoles; n++)
    for (i = 0; i < dim; i++) {
      unsigned long k;
      for (k = length - 1; k > 0; k--)
	series[i][k] -= series[i][k - 1];
    }

  for (i = 0; i < dim; i++)
    series[i] += ipoles;
  length -= ipoles;

  /* set_averages_to_zero(), with variance() reimplemented inline so a
     zero-variance component returns NULL instead of exiting the process. */
  check_alloc(average = (double *)malloc(sizeof(double) * dim));
  for (i = 0; i < dim; i++) {
    unsigned long k;
    double av = 0.0, var = 0.0, h;

    for (k = 0; k < length; k++) {
      h = series[i][k];
      av += h;
      var += h * h;
    }
    av /= (double)length;
    var = sqrt(fabs(var / (double)length - av * av));
    if (var == 0.0) {
      if (error != NULL)
	*error = ARIMA_MODEL_ERR_ZERO_VARIANCE;
      free(average);
      for (i = 0; i < dim; i++)
	free(series[i] - ipoles);
      free(series);
      return NULL;
    }
    average[i] = av;
    for (k = 0; k < length; k++)
      series[i][k] -= av;
  }

  if (poles < 1 || poles >= length ||
      (arimaset && (arpoles >= length || mapoles >= length))) {
    if (error != NULL)
      *error = ARIMA_MODEL_ERR_TOO_MANY_POLES;
    free(average);
    for (i = 0; i < dim; i++)
      free(series[i] - ipoles);
    free(series);
    return NULL;
  }

  /* Initial AR fit, order `poles`. */
  ardim = poles * dim;
  aindex = make_ar_index(dim, poles);

  check_alloc(vec = (double *)malloc(sizeof(double) * ardim));
  check_alloc(coeff = (double **)malloc(sizeof(double *) * dim));
  inverse = build_matrix((double *const *)series, length, poles, 0, aindex, ardim);
  for (i = 0; i < dim; i++) {
    build_vector(vec, ardim, i, (double *const *)series, length, poles, 0, aindex);
    coeff[i] = multiply_matrix_vector(inverse, vec, ardim);
  }
  for (i = 0; i < ardim; i++)
    free(inverse[i]);
  free(inverse);
  free(vec);

  check_alloc(diff = (double **)malloc(sizeof(double *) * dim));
  for (i = 0; i < dim; i++)
    check_alloc(diff[i] = (double *)calloc(length, sizeof(double)));

  check_alloc(rms_error = (double *)malloc(sizeof(double) * dim));
  make_residuals(diff, coeff, rms_error, (double *const *)series, length, dim,
		 poles, aindex, ardim);

  size = ardim;
  order = poles;

  if (arimaset) {
    unsigned int offset = poles;
    unsigned int combined_poles = (arpoles > mapoles) ? arpoles : mapoles;
    double **hseries, **oldcoeff;
    double **xdiff_full;
    double *diffcoeff_full;
    unsigned int iter;

    free_index(aindex);
    for (i = 0; i < dim; i++)
      free(coeff[i]);
    free(coeff);
    coeff = NULL;

    armadim = (arpoles + mapoles) * dim;
    aindex = make_arima_index(dim, arpoles, mapoles);
    size = armadim;

    check_alloc(hseries = (double **)malloc(sizeof(double *) * 2 * dim));
    for (i = 0; i < dim; i++) {
      check_alloc(hseries[i] = (double *)malloc(sizeof(double) * length));
      check_alloc(hseries[i + dim] = (double *)malloc(sizeof(double) * length));
      for (j = 0; j < length; j++) {
	hseries[i][j] = series[i][j];
	hseries[i + dim][j] = diff[i][j];
      }
    }
    for (i = 0; i < dim; i++)
      free(series[i] - ipoles);
    free(series);
    series = hseries;

    check_alloc(oldcoeff = (double **)malloc(sizeof(double *) * dim));
    for (i = 0; i < dim; i++) {
      check_alloc(oldcoeff[i] = (double *)malloc(sizeof(double) * armadim));
      for (j = 0; j < armadim; j++)
	oldcoeff[i][j] = 0.0;
    }

    check_alloc(xdiff_full = (double **)malloc(sizeof(double *) * iterations));
    for (i = 0; i < iterations; i++)
      check_alloc(xdiff_full[i] = (double *)malloc(sizeof(double) * dim));
    check_alloc(diffcoeff_full = (double *)malloc(sizeof(double) * iterations));

    for (iter = 1; iter <= iterations; iter++) {
      double **imat_inverse, *ivec, **icoeff;
      double hdiff, alldiff;
      unsigned int hj;

      check_alloc(ivec = (double *)malloc(sizeof(double) * armadim));
      check_alloc(icoeff = (double **)malloc(sizeof(double *) * dim));

      offset += combined_poles;
      imat_inverse = build_matrix((double *const *)series, length,
				   combined_poles, offset, aindex, armadim);
      for (i = 0; i < dim; i++) {
	build_vector(ivec, armadim, i, (double *const *)series, length,
		     combined_poles, offset, aindex);
	icoeff[i] = multiply_matrix_vector(imat_inverse, ivec, armadim);
      }
      for (i = 0; i < armadim; i++)
	free(imat_inverse[i]);
      free(imat_inverse);
      free(ivec);

      make_residuals(diff, icoeff, rms_error, (double *const *)series, length,
		      dim, combined_poles, aindex, armadim);

      for (j = 0; j < dim; j++) {
	hdiff = 0.0;
	hj = j + dim;
	for (i = offset; i < length; i++)
	  hdiff += sqr(series[hj][i] - diff[j][i]);
	for (i = 0; i < length; i++)
	  series[hj][i] = diff[j][i];
	xdiff_full[iter - 1][j] = sqrt(hdiff / (double)(length - offset));
      }

      diffcoeff_full[iter - 1] = 0.0;
      for (i = 0; i < dim; i++)
	for (j = 0; j < dim; j++) {
	  /* Mirrors the original's dim-only (not armadim-only) comparison of
	     the coefficient matrix - not every coefficient, just the leading
	     dim x dim block. Preserved as-is for byte-identical CLI
	     convergence behaviour, odd as it looks. */
	  diffcoeff_full[iter - 1] += sqr(icoeff[i][j] - oldcoeff[i][j]);
	  oldcoeff[i][j] = icoeff[i][j];
	}
      diffcoeff_full[iter - 1] = sqrt(diffcoeff_full[iter - 1] / (double)armadim);

      alldiff = xdiff_full[iter - 1][0];
      for (i = 1; i < dim; i++)
	if (xdiff_full[iter - 1][i] > alldiff)
	  alldiff = xdiff_full[iter - 1][i];
      realiter = iter;
      if (alldiff < convergence)
	iter = iterations;

      if (iter < iterations) {
	for (i = 0; i < dim; i++)
	  free(icoeff[i]);
	free(icoeff);
      }
      else
	coeff = icoeff;
    }

    for (i = 0; i < dim; i++)
      free(oldcoeff[i]);
    free(oldcoeff);

    order = combined_poles;

    check_alloc(xdiff_out = (double **)malloc(sizeof(double *) * realiter));
    for (i = 0; i < realiter; i++) {
      check_alloc(xdiff_out[i] = (double *)malloc(sizeof(double) * dim));
      for (j = 0; j < dim; j++)
	xdiff_out[i][j] = xdiff_full[i][j];
    }
    check_alloc(diffcoeff_out = (double *)malloc(sizeof(double) * realiter));
    for (i = 0; i < realiter; i++)
      diffcoeff_out[i] = diffcoeff_full[i];

    for (i = 0; i < iterations; i++)
      free(xdiff_full[i]);
    free(xdiff_full);
    free(diffcoeff_full);
  }

  check_alloc(model = (ARIMAModel *)malloc(sizeof(ARIMAModel)));
  model->dim = dim;
  model->length = length;
  model->poles = poles;
  model->arpoles = arpoles;
  model->ipoles = ipoles;
  model->mapoles = mapoles;
  model->arimaset = arimaset;
  model->order = order;
  model->size = size;
  model->average = average;
  model->series = series;
  model->coeff = coeff;
  model->rms_error = rms_error;
  model->residuals = diff;
  model->aindex = aindex;
  model->realiter = realiter;
  model->xdiff = xdiff_out;
  model->diffcoeff = diffcoeff_out;

  return model;
}

void arima_model_free(ARIMAModel *model)
{
  unsigned int i;

  if (model == NULL)
    return;

  if (model->arimaset) {
    for (i = 0; i < 2 * model->dim; i++)
      free(model->series[i]);
  }
  else {
    for (i = 0; i < model->dim; i++)
      free(model->series[i] - model->ipoles);
  }
  free(model->series);

  for (i = 0; i < model->dim; i++) {
    free(model->coeff[i]);
    free(model->residuals[i]);
  }
  free(model->coeff);
  free(model->residuals);
  free(model->rms_error);
  free(model->average);

  free_index(model->aindex);

  if (model->arimaset) {
    for (i = 0; i < model->realiter; i++)
      free(model->xdiff[i]);
    free(model->xdiff);
    free(model->diffcoeff);
  }

  free(model);
}

static double **iterate_ar(const ARIMAModel *model, unsigned long ilength,
			    unsigned long seed)
{
  unsigned int dim = model->dim, poles = model->order;
  long i, j, i1, i2, n, d;
  double **iterate, *swap, **myrand, **out;

  check_alloc(iterate = (double **)malloc(sizeof(double *) * (poles + 1)));
  for (i = 0; i <= poles; i++)
    check_alloc(iterate[i] = (double *)malloc(sizeof(double) * dim));

  check_alloc(myrand = (double **)malloc(sizeof(double *) * dim));
  for (i = 0; i < dim; i++)
    myrand[i] = rand_arb_dist(model->residuals[i], model->length,
			       ilength + poles, 100, seed);

  rnd_init(seed);
  for (i = 0; i < 1000; i++)
    rnd_long();

  for (i = 0; i < dim; i++)
    for (j = 0; j < poles; j++)
      iterate[j][i] = myrand[i][j];

  check_alloc(out = (double **)malloc(sizeof(double *) * ilength));

  for (n = 0; n < (long)ilength; n++) {
    for (d = 0; d < dim; d++) {
      iterate[poles][d] = myrand[d][n + poles];
      for (i1 = 0; i1 < dim; i1++)
	for (i2 = 0; i2 < poles; i2++)
	  iterate[poles][d] +=
	      model->coeff[d][i1 * poles + i2] * iterate[poles - 1 - i2][i1];
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
  for (i = 0; i < dim; i++)
    free(myrand[i]);
  free(myrand);

  return out;
}

static double **iterate_arma(const ARIMAModel *model, unsigned long ilength,
			      unsigned long seed)
{
  unsigned int dim = model->dim, poles = model->order, armadim = model->size;
  unsigned int **aindex = model->aindex;
  long i, j, n, d;
  long is, id;
  double **iterate, *swap, **myrand, **out;

  check_alloc(iterate = (double **)malloc(sizeof(double *) * (poles + 1)));
  for (i = 0; i <= poles; i++)
    check_alloc(iterate[i] = (double *)malloc(sizeof(double) * 2 * dim));

  check_alloc(myrand = (double **)malloc(sizeof(double *) * dim));
  for (i = 0; i < dim; i++)
    myrand[i] = rand_arb_dist(model->residuals[i], model->length,
			       ilength + poles, 100, seed);

  rnd_init(seed);
  for (i = 0; i < 1000; i++)
    rnd_long();

  for (i = 0; i < dim; i++)
    for (j = 0; j < poles; j++)
      iterate[j][i] = iterate[j][dim + i] = myrand[i][j];

  check_alloc(out = (double **)malloc(sizeof(double *) * ilength));

  for (n = 0; n < (long)ilength; n++) {
    for (i = 0; i < dim; i++)
      iterate[poles][i] = iterate[poles][i + dim] = myrand[i][n + poles];

    for (j = 0; j < dim; j++)
      for (i = 0; i < armadim; i++) {
	id = aindex[0][i];
	is = aindex[1][i];
	iterate[poles][j] += model->coeff[j][i] * iterate[poles - 1 - is][id];
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
  for (i = 0; i < dim; i++)
    free(myrand[i]);
  free(myrand);

  return out;
}

double **arima_model_iterate(const ARIMAModel *model, unsigned long ilength,
			      unsigned long seed)
{
  if (model->arimaset)
    return iterate_arma(model, ilength, seed);
  return iterate_ar(model, ilength, seed);
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
