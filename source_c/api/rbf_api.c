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

/* Reentrant core of rbf, factored out of source_c/rbf.c so it has no
   dependency on argv parsing, file-scope globals, or the process-exiting
   error paths in the generic variance()/rescale_data()/solvele() library
   routines it used to call. The math here (avdistance/rbf-kernel/drift/
   make_fit/forecast_error/make_cast) is unchanged from the original
   main(): rescale_data()'s min/max scan (whose interval == 0 check also
   stands in for the immediately-following variance() call's zero-variance
   check - see the comment below) and solvele()'s Gaussian elimination are
   reimplemented inline here so their error conditions become a NULL
   return plus an RBFError code instead. */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../routines/tsa.h"
#include "../../include/rbf.h"

static double rbf_kernel(const double *act, const double *cen,
			  unsigned int dim, unsigned int delay, double varianz)
{
  double denum = 2.0 * varianz * varianz;
  double r = 0.0;
  unsigned int i;

  for (i = 0; i < dim; i++)
    r += sqr(*(act - i * delay) - cen[i]);

  return exp(-r / denum);
}

static double avdistance(double *const *center, unsigned int centers,
			  unsigned int dim)
{
  unsigned int i, j, k;
  double dist = 0.0;

  for (i = 0; i < centers; i++)
    for (j = 0; j < centers; j++)
      if (i != j)
	for (k = 0; k < dim; k++)
	  dist += sqr(center[i][k] - center[j][k]);

  return sqrt(dist / (centers - 1) / centers / dim);
}

static void apply_drift(double **center, unsigned int centers, unsigned int dim)
{
  double *force, h, h1, stepsize = 1e-2, step1;
  unsigned int i, j, k, l;

  check_alloc(force = (double *)malloc(sizeof(double) * dim));
  for (l = 0; l < 20; l++) {
    for (i = 0; i < centers; i++) {
      for (j = 0; j < dim; j++) {
	force[j] = 0.0;
	for (k = 0; k < centers; k++) {
	  if (k != i) {
	    h = center[i][j] - center[k][j];
	    force[j] += h / sqr(h) / fabs(h);
	  }
	}
      }
      h = 0.0;
      for (j = 0; j < dim; j++)
	h += sqr(force[j]);
      step1 = stepsize / sqrt(h);
      for (j = 0; j < dim; j++) {
	h1 = step1 * force[j];
	if (((center[i][j] + h1) > -0.1) && ((center[i][j] + h1) < 1.1))
	  center[i][j] += h1;
      }
    }
  }
  free(force);
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

/* Returns 1 if the fit's normal-equations matrix was singular (coefs is
   left untouched in that case), 0 on success. */
static int make_fit_core(const double *resc, double *const *center,
			  unsigned int centers, unsigned int dim,
			  unsigned int delay, double varianz,
			  unsigned long step, unsigned long insample,
			  double *coefs)
{
  double **mat, *hcen;
  double h;
  unsigned long n, nst;
  unsigned int i, j;
  int singular;

  check_alloc(mat = (double **)malloc(sizeof(double *) * (centers + 1)));
  for (i = 0; i <= centers; i++)
    check_alloc(mat[i] = (double *)malloc(sizeof(double) * (centers + 1)));
  check_alloc(hcen = (double *)malloc(sizeof(double) * centers));

  for (i = 0; i <= centers; i++) {
    coefs[i] = 0.0;
    for (j = 0; j <= centers; j++)
      mat[i][j] = 0.0;
  }

  for (n = (unsigned long)(dim - 1) * delay; n < insample - step; n++) {
    nst = n + step;
    for (i = 0; i < centers; i++)
      hcen[i] = rbf_kernel(&resc[n], center[i], dim, delay, varianz);
    coefs[0] += resc[nst];
    mat[0][0] += 1.0;
    for (i = 1; i <= centers; i++)
      mat[i][0] += hcen[i - 1];
    for (i = 1; i <= centers; i++) {
      coefs[i] += resc[nst] * (h = hcen[i - 1]);
      for (j = 1; j <= i; j++)
	mat[i][j] += h * hcen[j - 1];
    }
  }

  h = (double)(insample - step - (unsigned long)(dim - 1) * delay);
  for (i = 0; i <= centers; i++) {
    coefs[i] /= h;
    for (j = 0; j <= i; j++) {
      mat[i][j] /= h;
      mat[j][i] = mat[i][j];
    }
  }

  singular = solvele_core(mat, coefs, (unsigned int)(centers + 1));

  for (i = 0; i <= centers; i++)
    free(mat[i]);
  free(mat);
  free(hcen);

  return singular;
}

static double forecast_error(const double *resc, double *const *center,
			      const double *coefs, unsigned int centers,
			      unsigned int dim, unsigned int delay,
			      double varianz, unsigned long step,
			      unsigned long i0, unsigned long i1)
{
  unsigned int i;
  unsigned long n;
  double h, error = 0.0;

  for (n = i0 + (unsigned long)(dim - 1) * delay; n < i1 - step; n++) {
    h = coefs[0];
    for (i = 1; i <= centers; i++)
      h += coefs[i] * rbf_kernel(&resc[n], center[i - 1], dim, delay, varianz);
    error += (resc[n + step] - h) * (resc[n + step] - h);
  }

  return sqrt(error / (double)(i1 - i0 - step - (unsigned long)(dim - 1) * delay));
}

static double *make_cast(const double *resc, unsigned long length,
			  double *const *center, const double *coefs,
			  unsigned int centers, unsigned int dim,
			  unsigned int delay, double varianz,
			  double min, double interval, unsigned long cast_length)
{
  double *cast, new_el, *out;
  unsigned long n;
  unsigned int i, windim = (dim - 1) * delay;

  check_alloc(cast = (double *)malloc(sizeof(double) * (windim + 1)));
  for (i = 0; i <= windim; i++)
    cast[i] = resc[length - 1 - windim + i];

  check_alloc(out = (double *)malloc(sizeof(double) * cast_length));

  for (n = 0; n < cast_length; n++) {
    new_el = coefs[0];
    for (i = 1; i <= centers; i++)
      new_el += coefs[i] * rbf_kernel(&cast[windim], center[i - 1], dim, delay, varianz);
    out[n] = new_el * interval + min;
    for (i = 0; i < windim; i++)
      cast[i] = cast[i + 1];
    cast[windim] = new_el;
  }

  free(cast);
  return out;
}

RBFResult *rbf_fit(const double *series, unsigned long length,
		    unsigned int dim, unsigned int delay, unsigned int centers,
		    int drift, unsigned long step, unsigned long insample,
		    unsigned long cast_length, RBFError *error)
{
  unsigned long i, effective_insample;
  unsigned int effective_centers, j;
  long cstep;
  double min, interval, av, sigma;
  double varianz;
  double *resc;
  double **center;
  double *coefs;
  RBFResult *result;

  if (error != NULL)
    *error = RBF_OK;

  /* rescale_data()'s min/max scan. interval == 0 (data is constant) is the
     only condition under which the CLI's rescale_data()/variance() calls
     can exit(): rescale_data() runs first and always catches this case,
     so variance()'s own zero-variance check (on the now-rescaled data)
     can never additionally trigger - a successfully rescaled series always
     has at least one point at 0.0 and one at 1.0. */
  min = interval = series[0];
  for (i = 1; i < length; i++) {
    if (series[i] < min)
      min = series[i];
    if (series[i] > interval)
      interval = series[i];
  }
  interval -= min;
  if (interval == 0.0) {
    if (error != NULL)
      *error = RBF_ERR_ZERO_VARIANCE;
    return NULL;
  }

  check_alloc(resc = (double *)malloc(sizeof(double) * length));
  for (i = 0; i < length; i++)
    resc[i] = (series[i] - min) / interval;

  effective_insample = insample > length ? length : insample;
  effective_centers = centers > (unsigned long)length ? (unsigned int)length : centers;

  check_alloc(coefs = (double *)malloc(sizeof(double) * (effective_centers + 1)));
  check_alloc(center = (double **)malloc(sizeof(double *) * effective_centers));
  for (i = 0; i < effective_centers; i++)
    check_alloc(center[i] = (double *)malloc(sizeof(double) * dim));

  cstep = (long)length - 1 - (long)(dim - 1) * delay;
  for (i = 0; i < effective_centers; i++)
    for (j = 0; j < dim; j++)
      center[i][j] = resc[(long)(dim - 1) * delay - (long)j * delay
			   + ((long)i * cstep) / ((long)effective_centers - 1)];

  if (drift)
    apply_drift(center, effective_centers, dim);
  varianz = avdistance((double *const *)center, effective_centers, dim);

  if (make_fit_core(resc, (double *const *)center, effective_centers, dim, delay,
		     varianz, step, effective_insample, coefs)) {
    for (i = 0; i < effective_centers; i++)
      free(center[i]);
    free(center);
    free(coefs);
    free(resc);
    if (error != NULL)
      *error = RBF_ERR_SINGULAR_MATRIX;
    return NULL;
  }

  check_alloc(result = (RBFResult *)malloc(sizeof(RBFResult)));
  result->dim = dim;
  result->delay = delay;
  result->centers = effective_centers;
  result->step = step;
  result->insample = effective_insample;
  result->length = length;

  check_alloc(result->center = (double **)malloc(sizeof(double *) * effective_centers));
  for (i = 0; i < effective_centers; i++) {
    check_alloc(result->center[i] = (double *)malloc(sizeof(double) * dim));
    for (j = 0; j < dim; j++)
      result->center[i][j] = center[i][j] * interval + min;
  }

  result->variance = varianz * interval;

  check_alloc(result->coefs = (double *)malloc(sizeof(double) * (effective_centers + 1)));
  result->coefs[0] = coefs[0] * interval + min;
  for (i = 1; i <= effective_centers; i++)
    result->coefs[i] = coefs[i] * interval;

  av = sigma = 0.0;
  for (i = 0; i < effective_insample; i++) {
    av += resc[i];
    sigma += resc[i] * resc[i];
  }
  av /= (double)effective_insample;
  sigma = sqrt(fabs(sigma / (double)effective_insample - av * av));
  result->insample_error =
    forecast_error(resc, (double *const *)center, coefs, effective_centers, dim, delay,
		   varianz, step, 0LU, effective_insample) / sigma;

  if (effective_insample < length) {
    av = sigma = 0.0;
    for (i = effective_insample; i < length; i++) {
      av += resc[i];
      sigma += resc[i] * resc[i];
    }
    av /= (double)(length - effective_insample);
    sigma = sqrt(fabs(sigma / (double)(length - effective_insample) - av * av));
    result->has_outsample_error = 1;
    result->outsample_error =
      forecast_error(resc, (double *const *)center, coefs, effective_centers, dim, delay,
		     varianz, step, effective_insample, length) / sigma;
  }
  else {
    result->has_outsample_error = 0;
    result->outsample_error = 0.0;
  }

  result->cast_length = cast_length;
  if (cast_length > 0)
    result->cast = make_cast(resc, length, (double *const *)center, coefs, effective_centers,
			      dim, delay, varianz, min, interval, cast_length);
  else
    result->cast = NULL;

  for (i = 0; i < effective_centers; i++)
    free(center[i]);
  free(center);
  free(coefs);
  free(resc);

  return result;
}

void rbf_free(RBFResult *result)
{
  unsigned int i;

  if (result == NULL)
    return;
  for (i = 0; i < result->centers; i++)
    free(result->center[i]);
  free(result->center);
  free(result->coefs);
  free(result->cast);
  free(result);
}
