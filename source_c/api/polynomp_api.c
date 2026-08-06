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

/* Reentrant core of polynomp, factored out of source_c/polynomp.c so it has
   no dependency on argv parsing, file-scope globals, or writing straight to
   a FILE*. The math here (polynom/make_fit/forecast_error/make_cast) is
   unchanged from the original main(): the two library routines it used to
   call that can exit() the process on bad input, variance() (zero-variance
   series) and solvele() (singular normal-equations matrix), are
   reimplemented inline here so their error conditions become a NULL return
   plus a PolynompError code instead. */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../routines/tsa.h"
#include "../../include/polynomp.h"

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
			  double *param)
{
  double **mat, *vec;
  double h;
  unsigned long n, hn;
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

  for (n = (unsigned long)((dim - 1) * delay); n < insample - 1; n++) {
    hn = n + 1;
    for (i = 0; i < plength; i++) {
      vec[i] += series[hn] * (h = polynom_term(series, order, dim, delay, n, i));
      for (j = i; j < plength; j++)
	mat[i][j] += polynom_term(series, order, dim, delay, n, j) * h;
    }
  }
  for (i = 0; i < plength; i++) {
    vec[i] /= (double)(insample - (unsigned long)((dim - 1) * delay) - 1);
    for (j = i; j < plength; j++)
      mat[j][i] = (mat[i][j] /=
		   (double)((insample - (unsigned long)((dim - 1) * delay)) - 1));
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
			      unsigned long i0, unsigned long i1)
{
  unsigned int i;
  unsigned long n;
  double h, error = 0.0;

  for (n = i0 + (unsigned long)((dim - 1) * delay); n < i1 - 1; n++) {
    h = 0.0;
    for (i = 0; i < plength; i++)
      h += param[i] * polynom_term(series, order, dim, delay, n, i);
    error += (series[n + 1] - h) * (series[n + 1] - h);
  }

  return sqrt(error / (double)(i1 - i0 - (unsigned long)((dim - 1) * delay) - 1));
}

/* Forecasts `step` points beyond series[0..length-1], the same way the
   CLI's make_cast() does it, except into a private scratch buffer seeded
   from (a copy of) the series' tail instead of overwriting series itself. */
static double *make_cast(const double *series, unsigned long length,
			  const unsigned int *order, unsigned int plength,
			  unsigned int dim, unsigned int delay,
			  const double *param, unsigned long step)
{
  double *window, *forecast;
  unsigned long i, j, hi;
  unsigned int k;
  double casted;

  hi = (unsigned long)((dim - 1) * delay);
  check_alloc(window = (double *)malloc(sizeof(double) * (hi + 1)));
  for (i = 0; i <= hi; i++)
    window[i] = series[length - hi + i - 1];

  check_alloc(forecast = (double *)malloc(sizeof(double) * (step ? step : 1)));
  for (i = 1; i <= step; i++) {
    casted = 0.0;
    for (k = 0; k < plength; k++)
      casted += param[k] * polynom_term(window, order, dim, delay, hi, k);
    forecast[i - 1] = casted;
    for (j = 0; j < hi; j++)
      window[j] = window[j + 1];
    window[hi] = casted;
  }

  free(window);
  return forecast;
}

PolynompResult *polynomp_fit(const double *series, unsigned long length,
			      const unsigned int *order, unsigned int plength,
			      unsigned int dim, unsigned int delay,
			      unsigned long insample, unsigned long step,
			      PolynompError *error)
{
  unsigned long i, effective_insample;
  int has_outsample;
  double av, varianz, h;
  double *param;
  PolynompResult *result;

  if (error != NULL)
    *error = POLYNOMP_OK;

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
      *error = POLYNOMP_ERR_ZERO_VARIANCE;
    return NULL;
  }

  if (insample >= length) {
    effective_insample = length;
    has_outsample = 0;
  }
  else {
    effective_insample = insample;
    has_outsample = 1;
  }

  check_alloc(param = (double *)malloc(sizeof(double) * plength));
  if (make_fit_core(series, order, plength, dim, delay, effective_insample, param)) {
    free(param);
    if (error != NULL)
      *error = POLYNOMP_ERR_SINGULAR_MATRIX;
    return NULL;
  }

  check_alloc(result = (PolynompResult *)malloc(sizeof(PolynompResult)));
  result->dim = dim;
  result->delay = delay;
  result->plength = plength;
  result->param = param;
  result->fce_insample =
    forecast_error(series, order, plength, dim, delay, param, 0LU, effective_insample)
    / varianz;
  result->has_outsample = has_outsample;
  result->fce_outsample = has_outsample
    ? forecast_error(series, order, plength, dim, delay, param,
		      effective_insample + 1, length) / varianz
    : 0.0;
  result->step = step;
  result->forecast =
    make_cast(series, length, order, plength, dim, delay, param, step);

  return result;
}

void polynomp_free(PolynompResult *result)
{
  if (result == NULL)
    return;
  free(result->param);
  free(result->forecast);
  free(result);
}
