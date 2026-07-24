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

/* Reentrant core of ar-model, factored out of source_c/ar-model.c so it
   has no dependency on argv parsing or file-scope globals. The math here
   is unchanged from the original build_matrix/build_vector/
   multiply_matrix_vector/make_residuals/iterate_model functions. */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../routines/tsa.h"
#include "../../include/ar-model.h"

static double **build_matrix(double *const *series, unsigned long length,
			      unsigned int dim, unsigned int poles)
{
  long n, i1, j1, i2, j2, hi, hj;
  double norm;
  double **mat;

  check_alloc(mat = (double **)malloc(sizeof(double *) * dim * poles));
  for (hi = 0; hi < (long)(dim * poles); hi++)
    check_alloc(mat[hi] = (double *)malloc(sizeof(double) * dim * poles));

  norm = 1. / ((double)length - (double)poles);

  for (i1 = 0; i1 < dim; i1++)
    for (i2 = 0; i2 < poles; i2++) {
      hi = i1 * poles + i2;
      for (j1 = 0; j1 < dim; j1++)
	for (j2 = 0; j2 < poles; j2++) {
	  hj = j1 * poles + j2;
	  mat[hi][hj] = 0.0;
	  for (n = poles - 1; n < (long)length - 1; n++)
	    mat[hi][hj] += series[i1][n - i2] * series[j1][n - j2];
	  mat[hi][hj] *= norm;
	}
    }

  return mat;
}

static void build_vector(double *vec, long comp, double *const *series,
			  unsigned long length, unsigned int dim,
			  unsigned int poles)
{
  long i1, i2, hi, n;
  double norm;

  norm = 1. / ((double)length - (double)poles);

  for (i1 = 0; i1 < (long)(poles * dim); i1++)
    vec[i1] = 0.0;

  for (i1 = 0; i1 < dim; i1++)
    for (i2 = 0; i2 < poles; i2++) {
      hi = i1 * poles + i2;
      for (n = poles - 1; n < (long)length - 1; n++)
	vec[hi] += series[comp][n + 1] * series[i1][n - i2];
      vec[hi] *= norm;
    }
}

static double *multiply_matrix_vector(double **mat, double *vec,
				       unsigned int dim, unsigned int poles)
{
  long i, j;
  double *new_vec;

  check_alloc(new_vec = (double *)malloc(sizeof(double) * poles * dim));

  for (i = 0; i < (long)(poles * dim); i++) {
    new_vec[i] = 0.0;
    for (j = 0; j < (long)(poles * dim); j++)
      new_vec[i] += mat[i][j] * vec[j];
  }
  return new_vec;
}

static void make_residuals(double **diff, double **coeff,
			    double *rms_error, double *const *series,
			    unsigned long length, unsigned int dim,
			    unsigned int poles)
{
  long n, d, i, j;

  for (i = 0; i < dim; i++)
    rms_error[i] = 0.0;

  for (n = poles - 1; n < (long)length - 1; n++) {
    for (d = 0; d < dim; d++) {
      diff[d][n + 1] = series[d][n + 1];
      for (i = 0; i < dim; i++)
	for (j = 0; j < poles; j++)
	  diff[d][n + 1] -= coeff[d][i * poles + j] * series[i][n - j];
      rms_error[d] += sqr(diff[d][n + 1]);
    }
  }
  for (i = 0; i < dim; i++)
    rms_error[i] = sqrt(rms_error[i] / ((double)length - (double)poles));
}

ARModel *ar_model_fit(double *const *series, unsigned long length,
		       unsigned int dim, unsigned int poles)
{
  long i, j;
  double *vec, **mat, **inverse;
  ARModel *model;

  if (poles < 1 || poles >= length)
    return NULL;

  check_alloc(vec = (double *)malloc(sizeof(double) * poles * dim));
  mat = build_matrix(series, length, dim, poles);
  inverse = invert_matrix(mat, (unsigned int)(dim * poles));

  check_alloc(model = (ARModel *)malloc(sizeof(ARModel)));
  model->dim = dim;
  model->poles = poles;
  model->length = length;
  check_alloc(model->coeff = (double **)malloc(sizeof(double *) * dim));
  check_alloc(model->rms_error = (double *)malloc(sizeof(double) * dim));
  check_alloc(model->residuals = (double **)malloc(sizeof(double *) * dim));
  for (i = 0; i < dim; i++) {
    build_vector(vec, i, series, length, dim, poles);
    model->coeff[i] = multiply_matrix_vector(inverse, vec, dim, poles);
    /* calloc, not malloc: entries [0..poles-1] are never written by
       make_residuals() below (only [poles..length-1] are one-step-ahead
       residuals - see ar-model.h), and callers like the Python bindings
       expose the whole [dim][length] array, so those entries must be
       deterministic rather than uninitialized heap memory. */
    check_alloc(model->residuals[i] = (double *)calloc(length, sizeof(double)));
  }

  make_residuals(model->residuals, model->coeff, model->rms_error,
		 series, length, dim, poles);

  free(vec);
  for (i = 0; i < (long)(dim * poles); i++) {
    free(mat[i]);
    free(inverse[i]);
  }
  free(mat);
  free(inverse);

  return model;
}

void ar_model_free(ARModel *model)
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
  free(model);
}

double **ar_model_iterate(const ARModel *model, unsigned long ilength,
			   unsigned long seed)
{
  long i, j, i1, i2, n, d;
  unsigned int dim = model->dim, poles = model->poles;
  double **iterate, *swap, **out;

  check_alloc(iterate = (double **)malloc(sizeof(double *) * (poles + 1)));
  for (i = 0; i <= poles; i++)
    check_alloc(iterate[i] = (double *)malloc(sizeof(double) * dim));
  rnd_init(seed);
  for (i = 0; i < 1000; i++)
    gaussian(1.0);
  for (i = 0; i < dim; i++)
    for (j = 0; j < poles; j++)
      iterate[j][i] = gaussian(model->rms_error[i]);

  check_alloc(out = (double **)malloc(sizeof(double *) * ilength));

  for (n = 0; n < (long)ilength; n++) {
    for (d = 0; d < dim; d++) {
      iterate[poles][d] = gaussian(model->rms_error[d]);
      for (i1 = 0; i1 < dim; i1++)
	for (i2 = 0; i2 < poles; i2++)
	  iterate[poles][d] += model->coeff[d][i1 * poles + i2]
	    * iterate[poles - 1 - i2][i1];
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

  return out;
}

void ar_model_iterate_free(double **iterated, unsigned long ilength)
{
  unsigned long n;

  if (iterated == NULL)
    return;
  for (n = 0; n < ilength; n++)
    free(iterated[n]);
  free(iterated);
}
