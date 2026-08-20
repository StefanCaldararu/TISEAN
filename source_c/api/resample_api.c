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

/* Reentrant core of resample, factored out of source_c/resample.c so it
   has no dependency on argv parsing or file-scope globals. The
   interpolation math (build the (order+1)x(order+1) Vandermonde-like
   matrix, invert it, then evaluate the local polynomial fit at each new
   grid point) is unchanged from main(); invert_matrix()/solvele() are
   reimplemented inline below (resample_invert/resample_solvele) so that a
   singular matrix returns NULL instead of exiting the process. */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../routines/tsa.h"
#include "../../include/resample.h"

#define RESAMPLE_INITIAL_CAPACITY 1024

/* Mirrors solvele()'s Gaussian elimination with partial pivoting exactly,
   except it returns 0 instead of printing a message and exiting when a
   pivot is found to be zero. mat and vec are modified in place. */
static int resample_solvele(double **mat, double *vec, unsigned int n)
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
      return 0;
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
  return 1;
}

/* Mirrors invert_matrix() exactly, except it returns NULL (freeing
   everything it allocated) instead of exiting the process when
   resample_solvele() reports a singular matrix. */
static double **resample_invert(double **mat, unsigned int size)
{
  int i, j, k;
  double **hmat, **imat, *vec;

  check_alloc(hmat = (double **)malloc(sizeof(double *) * size));
  for (i = 0; i < (int)size; i++)
    check_alloc(hmat[i] = (double *)malloc(sizeof(double) * size));

  check_alloc(imat = (double **)malloc(sizeof(double *) * size));
  for (i = 0; i < (int)size; i++)
    check_alloc(imat[i] = (double *)malloc(sizeof(double) * size));

  check_alloc(vec = (double *)malloc(sizeof(double) * size));

  for (i = 0; i < (int)size; i++) {
    for (j = 0; j < (int)size; j++) {
      vec[j] = (i == j) ? 1.0 : 0.0;
      for (k = 0; k < (int)size; k++)
	hmat[j][k] = mat[j][k];
    }
    if (!resample_solvele(hmat, vec, size)) {
      free(vec);
      for (j = 0; j < (int)size; j++)
	free(hmat[j]);
      free(hmat);
      for (j = 0; j < (int)size; j++)
	free(imat[j]);
      free(imat);
      return NULL;
    }
    for (j = 0; j < (int)size; j++)
      imat[j][i] = vec[j];
  }

  free(vec);
  for (i = 0; i < (int)size; i++)
    free(hmat[i]);
  free(hmat);

  return imat;
}

ResampleResult *resample_compute(const double *series, unsigned long length,
				  double sampletime, unsigned int order)
{
  long i, j, itime, itime_old;
  int horder, horder2;
  double **mat, *vec, **imat, *coef;
  double time, htime, new_el;
  unsigned long out_len, out_cap;
  double *out;
  ResampleResult *result;

  horder = order + 1;
  horder2 = (horder + 1) / 2 - horder;

  check_alloc(mat = (double **)malloc(sizeof(double *) * horder));
  for (i = 0; i < horder; i++)
    check_alloc(mat[i] = (double *)malloc(sizeof(double) * horder));
  check_alloc(vec = (double *)malloc(sizeof(double) * horder));
  check_alloc(coef = (double *)malloc(sizeof(double) * horder));

  for (i = 0; i < horder; i++)
    for (j = 0; j < horder; j++)
      mat[i][j] = pow((double)(horder2 + i), (double)j);

  imat = resample_invert(mat, (unsigned int)horder);

  for (i = 0; i < horder; i++)
    free(mat[i]);
  free(mat);

  if (imat == NULL) {
    free(vec);
    free(coef);
    return NULL;
  }

  out_cap = RESAMPLE_INITIAL_CAPACITY;
  check_alloc(out = (double *)malloc(sizeof(double) * out_cap));
  out_len = 0;

  time = (horder + 1) / 2.;
  itime_old = -1;
  while (time < (double)(length - horder / 2)) {
    itime = (int)time + horder2;
    if (itime != itime_old) {
      for (i = 0; i < horder; i++)
	vec[i] = series[i + itime];
      for (i = 0; i < horder; i++) {
	coef[i] = 0.0;
	for (j = 0; j < horder; j++)
	  coef[i] += imat[i][j] * vec[j];
      }
    }
    itime_old = itime;
    htime = time - itime + horder2;
    new_el = coef[0];
    for (i = 1; i < horder; i++)
      new_el += coef[i] * pow(htime, (double)i);

    if (out_len == out_cap) {
      out_cap *= 2;
      check_alloc(out = (double *)realloc(out, sizeof(double) * out_cap));
    }
    out[out_len++] = new_el;

    time += sampletime;
  }

  for (i = 0; i < horder; i++)
    free(imat[i]);
  free(imat);
  free(vec);
  free(coef);

  check_alloc(result = (ResampleResult *)malloc(sizeof(ResampleResult)));
  if (out_len > 0)
    check_alloc(out = (double *)realloc(out, sizeof(double) * out_len));
  result->length = out_len;
  result->data = out;

  return result;
}

void resample_free(ResampleResult *result)
{
  if (result == NULL)
    return;
  free(result->data);
  free(result);
}
