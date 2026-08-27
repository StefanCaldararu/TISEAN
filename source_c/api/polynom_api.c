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

/* Reentrant core of polynom, factored out of source_c/polynom.c so it has no
   dependency on argv parsing, file-scope globals, or writing straight to a
   FILE*. The math here (the polynom()/number_pars()/make_coding()/decode()/
   make_fit()/make_error()/make_cast() term-encoding and fit) is unchanged
   from the original main(): only variance(), the one library routine it used
   to call that can exit() the process on bad input (a zero-variance series),
   is reimplemented inline here so that error condition becomes a NULL return
   plus a PolynomError code instead. solvele() (a singular normal-equations
   matrix) is still the shared, process-exiting library routine, exactly like
   the original. */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../routines/tsa.h"
#include "../../include/polynom.h"

/* Evaluates one polynomial term at time index act: the product, over the
   `termdim` most recent points of the (delay-)embedding starting at act, of
   series[act - (termdim-1)*delay + ...] raised to the powers encoded by
   `cur`/`fac` - the exact recursive decoding make_coding() below encodes.
   termdim starts at dim and counts down to 1 as the recursion unwinds. */
static double poly_term(const double *series, unsigned int order,
			 unsigned int delay, int act, int termdim,
			 long cur, long fac)
{
  int j, n, hi;
  double ret = 1.0;

  n = cur / fac;
  hi = act - (termdim - 1) * (int)delay;
  for (j = 1; j <= n; j++)
    ret *= series[hi];
  if (termdim > 1)
    ret *= poly_term(series, order, delay, act, termdim - 1,
		      cur - (long)n * fac, fac / ((long)order + 1));

  return ret;
}

static int number_pars(unsigned int dim, int ord, int start)
{
  int i, ret = 0;

  if (ord == 1)
    for (i = start; i <= (int)dim; i++)
      ret += 1;
  else
    for (i = start; i <= (int)dim; i++)
      ret += number_pars(dim, ord - 1, i);

  return ret;
}

static void make_coding(long *coding, unsigned int *hpar, unsigned int order,
			 int ord, int d, long fac, long cur)
{
  int j;

  if (d == -1)
    coding[(*hpar)++] = cur;
  else
    for (j = 0; j <= ord; j++)
      make_coding(coding, hpar, order, ord - j, d - 1,
		  fac * ((long)order + 1), cur + (long)j * fac);
}

static void decode_term(int *out, unsigned int order, int level, long cur, long fac)
{
  int n;

  n = cur / fac;
  out[level] = n;
  if (level > 0)
    decode_term(out, order, level - 1, cur - (long)n * fac, fac / ((long)order + 1));
}

static void make_fit(const double *series, unsigned int dim, unsigned int delay,
		      unsigned int order, unsigned long insample,
		      const long *coding, long maxencode, unsigned int pars,
		      double *results)
{
  unsigned int i, j;
  long k;
  double **mat, *b;

  check_alloc(b = (double *)malloc(sizeof(double) * pars));
  check_alloc(mat = (double **)malloc(sizeof(double *) * pars));
  for (i = 0; i < pars; i++)
    check_alloc(mat[i] = (double *)malloc(sizeof(double) * pars));

  for (i = 0; i < pars; i++) {
    b[i] = 0.0;
    for (j = 0; j < pars; j++)
      mat[i][j] = 0.0;
  }

  for (i = 0; i < pars; i++)
    for (j = i; j < pars; j++)
      for (k = (long)(dim - 1) * delay; k < (long)insample - 1; k++)
	mat[i][j] += poly_term(series, order, delay, (int)k, (int)dim, coding[i], maxencode) *
	  poly_term(series, order, delay, (int)k, (int)dim, coding[j], maxencode);
  for (i = 0; i < pars; i++)
    for (j = i; j < pars; j++)
      mat[j][i] = (mat[i][j] /=
		   ((double)insample - 1.0 - (double)(dim - 1) * delay));

  for (i = 0; i < pars; i++) {
    for (k = (long)(dim - 1) * delay; k < (long)insample - 1; k++)
      b[i] += series[k + 1] *
	poly_term(series, order, delay, (int)k, (int)dim, coding[i], maxencode);
    b[i] /= ((double)insample - 1.0 - (double)(dim - 1) * delay);
  }

  solvele(mat, b, pars);

  for (i = 0; i < pars; i++)
    results[i] = b[i];

  free(b);
  for (i = 0; i < pars; i++)
    free(mat[i]);
  free(mat);
}

static double make_error(const double *series, unsigned int dim, unsigned int delay,
			  unsigned int order, const long *coding, long maxencode,
			  const double *results, unsigned int pars,
			  unsigned long i0, unsigned long i1)
{
  long j;
  unsigned int k;
  double h, err;

  err = 0.0;
  for (j = (long)i0 + (long)(dim - 1) * delay; j < (long)i1 - 1; j++) {
    h = 0.0;
    for (k = 0; k < pars; k++)
      h += results[k] * poly_term(series, order, delay, (int)j, (int)dim, coding[k], maxencode);
    err += (series[j + 1] - h) * (series[j + 1] - h);
  }
  return err / ((double)i1 - (double)i0 - (double)(dim - 1) * delay);
}

/* Forecasts `step` points beyond series[0..length-1], the same way the
   CLI's make_cast() does it, except into a private scratch buffer seeded
   from (a copy of) the series' tail instead of overwriting series itself,
   and converting back to original (unscaled) units via std_dev as it goes. */
static double *make_cast(const double *series, unsigned long length, unsigned int dim,
			  unsigned int delay, unsigned int order, const long *coding,
			  long maxencode, const double *results, unsigned int pars,
			  unsigned long step, double std_dev)
{
  double *window, *forecast, casted;
  unsigned long i, j, hi;
  unsigned int k;

  hi = (unsigned long)(dim - 1) * delay;
  check_alloc(window = (double *)malloc(sizeof(double) * (hi + 1)));
  for (i = 0; i <= hi; i++)
    window[i] = series[length - hi - 1 + i];

  check_alloc(forecast = (double *)malloc(sizeof(double) * (step ? step : 1)));
  for (i = 1; i <= step; i++) {
    casted = 0.0;
    for (k = 0; k < pars; k++)
      casted += results[k] *
	poly_term(window, order, delay, (int)hi, (int)dim, coding[k], maxencode);
    forecast[i - 1] = casted * std_dev;
    for (j = 0; j < hi; j++)
      window[j] = window[j + 1];
    window[hi] = casted;
  }

  free(window);
  return forecast;
}

PolynomResult *polynom_fit(const double *series, unsigned long length,
			    unsigned int dim, unsigned int delay,
			    unsigned int order, unsigned long insample,
			    unsigned long step, PolynomError *error)
{
  unsigned long i, effective_insample;
  int has_outsample;
  double av, std_dev, h;
  double *scaled;
  long *coding, maxencode;
  unsigned int hpar, pars, j, k, sumpar;
  int *opar;
  double *results, *coeff;
  int *exponent;
  PolynomResult *result;

  if (error != NULL)
    *error = POLYNOM_OK;

  av = std_dev = 0.0;
  for (i = 0; i < length; i++) {
    h = series[i];
    av += h;
    std_dev += h * h;
  }
  av /= (double)length;
  std_dev = sqrt(fabs(std_dev / (double)length - av * av));
  if (std_dev == 0.0) {
    if (error != NULL)
      *error = POLYNOM_ERR_ZERO_VARIANCE;
    return NULL;
  }

  check_alloc(scaled = (double *)malloc(sizeof(double) * length));
  for (i = 0; i < length; i++)
    scaled[i] = series[i] / std_dev;

  effective_insample = (insample >= length) ? length : insample;
  has_outsample = insample < length;

  maxencode = 1;
  for (i = 1; i < dim; i++)
    maxencode *= ((long)order + 1);

  pars = 1;
  for (i = 1; i <= order; i++)
    pars += (unsigned int)number_pars(dim, (int)i, 1);

  check_alloc(coding = (long *)malloc(sizeof(long) * pars));
  hpar = 0;
  make_coding(coding, &hpar, order, (int)order, (int)dim - 1, 1, 0);

  check_alloc(results = (double *)malloc(sizeof(double) * pars));
  make_fit(scaled, dim, delay, order, effective_insample, coding, maxencode, pars, results);

  check_alloc(opar = (int *)malloc(sizeof(int) * dim));
  check_alloc(coeff = (double *)malloc(sizeof(double) * pars));
  check_alloc(exponent = (int *)malloc(sizeof(int) * pars * dim));
  for (j = 0; j < pars; j++) {
    decode_term(opar, order, (int)dim - 1, coding[j], maxencode);
    sumpar = 0;
    for (k = 0; k < dim; k++) {
      sumpar += (unsigned int)opar[k];
      exponent[j * dim + k] = opar[k];
    }
    coeff[j] = results[j] / pow(std_dev, (double)((int)sumpar - 1));
  }

  check_alloc(result = (PolynomResult *)malloc(sizeof(PolynomResult)));
  result->dim = dim;
  result->delay = delay;
  result->order = order;
  result->plength = pars;
  result->coeff = coeff;
  result->exponent = exponent;

  result->error_insample =
    sqrt(make_error(scaled, dim, delay, order, coding, maxencode, results, pars,
		     0LU, effective_insample));

  result->has_outsample = has_outsample;
  result->error_outsample = has_outsample
    ? sqrt(make_error(scaled, dim, delay, order, coding, maxencode, results, pars,
		       effective_insample, length))
    : 0.0;

  result->step = step;
  result->forecast = step
    ? make_cast(scaled, length, dim, delay, order, coding, maxencode, results, pars,
		step, std_dev)
    : NULL;

  free(scaled);
  free(coding);
  free(results);
  free(opar);

  return result;
}

void polynom_free(PolynomResult *result)
{
  if (result == NULL)
    return;
  free(result->coeff);
  free(result->exponent);
  free(result->forecast);
  free(result);
}
