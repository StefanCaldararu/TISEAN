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

/* Reentrant core of polyback, factored out of source_c/polyback.c so it has
   no dependency on argv parsing, file-scope globals, or writing straight to
   a FILE*. The math here (polynom/make_fit/forecast_error and the backward
   elimination loop) is unchanged from the original main(): the two library
   routines it used to call that can exit() the process on bad input,
   variance() (zero-variance series) and solvele() (singular
   normal-equations matrix), are reimplemented inline here so their error
   conditions become a NULL return plus a PolybackError code instead. */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../routines/tsa.h"
#include "../../include/polyback.h"

static double polynom_term(const double *series, const unsigned int *order,
			    unsigned int dim, unsigned int delay,
			    unsigned long act, unsigned int which)
{
  unsigned int i, j;
  double ret = 1.0, h;

  for (i = 0; i < dim; i++) {
    h = series[act - (unsigned long)i * delay];
    for (j = 0; j < order[which * dim + i]; j++)
      ret *= h;
  }

  return ret;
}

/* Reimplements source_c/routines/solvele.c's Gaussian elimination with
   partial pivoting, but returns 1 instead of exit()ing when the matrix is
   singular. Otherwise byte-identical to the original. */
static int solvele_core(double **mat, double *vec, unsigned int n)
{
  double vswap, *mswap, *hvec, max, h, pivot, q;
  int i, j, k, maxi;

  for (i = 0; i < (int)n - 1; i++) {
    max = fabs(mat[i][i]);
    maxi = i;
    for (j = i + 1; j < (int)n; j++)
      if ((h = fabs(mat[j][i])) > max) {
	max = h;
	maxi = j;
      }
    if (maxi != i) {
      mswap = mat[i];
      mat[i] = mat[maxi];
      mat[maxi] = mswap;
      vswap = vec[i];
      vec[i] = vec[maxi];
      vec[maxi] = vswap;
    }

    hvec = mat[i];
    pivot = hvec[i];
    if (fabs(pivot) == 0.0)
      return 1;
    for (j = i + 1; j < (int)n; j++) {
      q = -mat[j][i] / pivot;
      mat[j][i] = 0.0;
      for (k = i + 1; k < (int)n; k++)
	mat[j][k] += q * hvec[k];
      vec[j] += q * vec[i];
    }
  }
  vec[n - 1] /= mat[n - 1][n - 1];
  for (i = (int)n - 2; i >= 0; i--) {
    hvec = mat[i];
    for (j = (int)n - 1; j > i; j--)
      vec[i] -= hvec[j] * vec[j];
    vec[i] /= hvec[i];
  }
  return 0;
}

/* Returns 1 if the fit's normal-equations matrix was singular (param is
   left untouched in that case), 0 on success. */
static int make_fit_core(const double *series, const unsigned int *order,
			  unsigned int plength, unsigned int dim,
			  unsigned int delay, unsigned long insample,
			  unsigned int step, double *param)
{
  double **mat, *vec;
  double h;
  unsigned long n;
  unsigned int i, j;
  int singular;

  check_alloc(vec = (double *)malloc(sizeof(double) * plength));
  check_alloc(mat = (double **)malloc(sizeof(double *) * plength));
  for (i = 0; i < plength; i++)
    check_alloc(mat[i] = (double *)malloc(sizeof(double) * plength));

  for (i = 0; i < plength; i++) {
    vec[i] = 0.0;
    for (j = 0; j < plength; j++)
      mat[i][j] = 0.0;
  }

  for (n = (unsigned long)(dim - 1) * delay; n < insample - step; n++) {
    for (i = 0; i < plength; i++) {
      vec[i] += series[n + step] * (h = polynom_term(series, order, dim, delay, n, i));
      for (j = i; j < plength; j++)
	mat[i][j] += polynom_term(series, order, dim, delay, n, j) * h;
    }
  }
  for (i = 0; i < plength; i++) {
    vec[i] /= (double)(insample - step - (unsigned long)(dim - 1) * delay);
    for (j = i; j < plength; j++)
      mat[j][i] = (mat[i][j] /=
		   (double)(insample - step - (unsigned long)(dim - 1) * delay));
  }

  singular = solvele_core(mat, vec, plength);
  if (!singular)
    for (i = 0; i < plength; i++)
      param[i] = vec[i];

  free(vec);
  for (i = 0; i < plength; i++)
    free(mat[i]);
  free(mat);

  return singular;
}

static double forecast_error(const double *series, const unsigned int *order,
			      unsigned int plength, unsigned int dim,
			      unsigned int delay, const double *param,
			      unsigned int step, unsigned long i0, unsigned long i1)
{
  unsigned int i;
  unsigned long n;
  double h, error = 0.0;

  for (n = i0 + (unsigned long)(dim - 1) * delay; n < i1 - step; n++) {
    h = 0.0;
    for (i = 0; i < plength; i++)
      h += param[i] * polynom_term(series, order, dim, delay, n, i);
    error += (series[n + step] - h) * (series[n + step] - h);
  }

  return sqrt(error / (double)(i1 - i0 - step - (unsigned long)(dim - 1) * delay));
}

PolybackResult *polyback_fit(const double *series, unsigned long length,
			      const unsigned int *order, unsigned long n_terms,
			      unsigned int dim, unsigned int delay,
			      unsigned long insample, unsigned int step,
			      unsigned int down_to, PolybackError *error)
{
  unsigned long i, n, effective_insample;
  unsigned int j, k, clamped_down_to, n_levels;
  int out_set;
  double av, varianz, h;
  double *full_param;
  unsigned int *isout;
  PolybackResult *result;

  if (error != NULL)
    *error = POLYBACK_OK;

  /* Reimplements variance()'s math (source_c/routines/variance.c), but
     returns NULL instead of exit()ing on a constant series. */
  av = varianz = 0.0;
  for (i = 0; i < length; i++) {
    h = series[i];
    av += h;
    varianz += h * h;
  }
  av /= (double)length;
  varianz = sqrt(fabs(varianz / (double)length - av * av));
  if (varianz == 0.0) {
    if (error != NULL)
      *error = POLYBACK_ERR_ZERO_VARIANCE;
    return NULL;
  }

  if (insample >= length) {
    effective_insample = length;
    out_set = 0;
  }
  else {
    effective_insample = insample;
    out_set = 1;
  }

  check_alloc(full_param = (double *)malloc(sizeof(double) * n_terms));
  if (make_fit_core(series, order, (unsigned int)n_terms, dim, delay,
		     effective_insample, step, full_param)) {
    free(full_param);
    if (error != NULL)
      *error = POLYBACK_ERR_SINGULAR_MATRIX;
    return NULL;
  }

  check_alloc(result = (PolybackResult *)malloc(sizeof(PolybackResult)));
  result->dim = dim;
  result->delay = delay;
  result->n_terms = n_terms;
  result->error_in =
    forecast_error(series, order, (unsigned int)n_terms, dim, delay, full_param,
		   step, 0LU, effective_insample) / varianz;
  result->has_outsample = out_set;
  result->error_out = out_set
    ? forecast_error(series, order, (unsigned int)n_terms, dim, delay, full_param,
		      step, effective_insample + 1, length) / varianz
    : 0.0;
  free(full_param);

  clamped_down_to = down_to;
  if (clamped_down_to < 1 || (unsigned long)clamped_down_to > n_terms)
    clamped_down_to = 1;
  n_levels = (unsigned int)(n_terms - clamped_down_to);
  result->n_levels = n_levels;

  result->level_n_terms = NULL;
  result->level_error_in = NULL;
  result->level_error_out = NULL;
  result->removed_index = NULL;
  if (n_levels > 0) {
    check_alloc(result->level_n_terms = (unsigned int *)malloc(sizeof(unsigned int) * n_levels));
    check_alloc(result->level_error_in = (double *)malloc(sizeof(double) * n_levels));
    check_alloc(result->level_error_out = (double *)malloc(sizeof(double) * n_levels));
    check_alloc(result->removed_index = (unsigned long *)malloc(sizeof(unsigned long) * n_levels));
  }

  check_alloc(isout = (unsigned int *)malloc(sizeof(unsigned int) * n_terms));
  for (i = 0; i < n_terms; i++)
    isout[i] = 0;

  for (n = 1; n <= n_levels; n++) {
    unsigned long plength = n_terms - n;
    unsigned long ibest = 0;
    double besti = 0.0, besto = 0.0, errori, erroro = 0.0;
    int have_best = 0;
    unsigned int *work_order;
    double *work_param;

    check_alloc(work_order = (unsigned int *)malloc(sizeof(unsigned int) * plength * dim));
    check_alloc(work_param = (double *)malloc(sizeof(double) * plength));

    for (i = 0; i < n_terms; i++) {
      if (!isout[i]) {
	unsigned long hl = 0;

	isout[i] = 1;
	for (k = 0; k < n_terms; k++) {
	  if (!isout[k]) {
	    for (j = 0; j < dim; j++)
	      work_order[hl * dim + j] = order[k * dim + j];
	    hl++;
	  }
	}

	if (make_fit_core(series, work_order, (unsigned int)plength, dim, delay,
			   effective_insample, step, work_param)) {
	  isout[i] = 0;
	  free(work_order);
	  free(work_param);
	  free(isout);
	  polyback_free(result);
	  if (error != NULL)
	    *error = POLYBACK_ERR_SINGULAR_MATRIX;
	  return NULL;
	}
	errori = forecast_error(series, work_order, (unsigned int)plength, dim,
				 delay, work_param, step, 0LU, effective_insample);
	if (out_set)
	  erroro = forecast_error(series, work_order, (unsigned int)plength, dim,
				   delay, work_param, step, effective_insample + 1,
				   length);

	if (!have_best) {
	  besti = errori;
	  if (out_set)
	    besto = erroro;
	  ibest = i;
	  have_best = 1;
	}
	else {
	  if (out_set) {
	    if (erroro < besto) {
	      besto = erroro;
	      besti = errori;
	      ibest = i;
	    }
	  }
	  else {
	    if (errori < besti) {
	      besti = errori;
	      besto = erroro;
	      ibest = i;
	    }
	  }
	}
	isout[i] = 0;
      }
    }
    isout[ibest] = 1;

    free(work_order);
    free(work_param);

    result->level_n_terms[n - 1] = (unsigned int)plength;
    result->level_error_in[n - 1] = besti / varianz;
    result->level_error_out[n - 1] = besto / varianz;
    result->removed_index[n - 1] = ibest;
  }

  free(isout);

  return result;
}

void polyback_free(PolybackResult *result)
{
  if (result == NULL)
    return;
  free(result->level_n_terms);
  free(result->level_error_in);
  free(result->level_error_out);
  free(result->removed_index);
  free(result);
}
