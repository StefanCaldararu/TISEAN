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

/* Reentrant core of mem_spec, factored out of source_c/mem_spec.c so it
   has no dependency on argv parsing, file-scope globals, or the
   process-exiting error path in the generic variance() library routine
   it used to call. The math here (mean/std via a plain sequential sum,
   Burg's recursion for the AR reflection coefficients in getcoefs(), and
   the transfer-function evaluation in powcoef()) is unchanged from
   main(). */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../routines/tsa.h"
#include "../../include/mem_spec.h"

#ifndef M_PI
#define M_PI 3.1415926535897932385E0
#endif

/* series is a caller-owned scratch buffer of `length` doubles (already
   mean-centered); it is mutated in place, mirroring the original
   getcoefs()'s in-place updates through its (self-advancing) global
   `series` pointer, but the pointer value handed to us is left
   untouched so the caller can still free() it afterwards. */
static double burg_coefs(double *series, unsigned long length,
			  unsigned long poles, double *coef)
{
  long i, j, hp = (long)poles - 1;
  double ret = 0.0, *cov, *help, *wseries, h1, h2;

  check_alloc(cov = (double *)malloc(sizeof(double) * length));
  check_alloc(help = (double *)malloc(sizeof(double) * poles));

  for (i = 0; i < (long)length; i++)
    ret += series[i] * series[i];
  ret /= (double)length;

  for (i = 0; i < (long)length; i++)
    cov[i] = series[i];
  wseries = series + 1;

  for (i = 0; i < (long)poles; i++) {
    h1 = h2 = 0.0;
    for (j = 0; j < (long)length - i - 1; j++) {
      h1 += cov[j] * wseries[j];
      h2 += cov[j] * cov[j] + wseries[j] * wseries[j];
    }
    coef[i] = 2.0 * h1 / h2;
    ret *= (1.0 - coef[i] * coef[i]);
    for (j = 0; j < i; j++)
      coef[j] = help[j] - coef[i] * help[i - 1 - j];
    if (i == hp)
      break;
    for (j = 0; j <= i; j++)
      help[j] = coef[j];
    for (j = 0; j < (long)length - i - 1; j++) {
      cov[j] -= help[i] * wseries[j];
      wseries[j] = wseries[j + 1] - help[i] * cov[j + 1];
    }
  }

  free(cov);
  free(help);

  return ret;
}

MemSpecModel *mem_spec_fit(const double *series, unsigned long length,
			    unsigned long poles)
{
  unsigned long i;
  double h, av, var, *centered, *coef;
  MemSpecModel *model;

  if (poles >= length)
    return NULL;

  av = var = 0.0;
  for (i = 0; i < length; i++) {
    h = series[i];
    av += h;
    var += h * h;
  }
  av /= (double)length;
  var = sqrt(fabs(var / (double)length - av * av));
  if (var == 0.0)
    return NULL;

  check_alloc(centered = (double *)malloc(sizeof(double) * length));
  for (i = 0; i < length; i++)
    centered[i] = series[i] - av;

  check_alloc(coef = (double *)malloc(sizeof(double) * poles));

  check_alloc(model = (MemSpecModel *)malloc(sizeof(MemSpecModel)));
  model->poles = poles;
  model->coef = coef;
  model->sigma2 = burg_coefs(centered, length, poles, coef);

  free(centered);

  return model;
}

void mem_spec_free(MemSpecModel *model)
{
  if (model == NULL)
    return;
  free(model->coef);
  free(model);
}

static double transfer_pow(double dt, const double *coef, unsigned long poles)
{
  unsigned long i;
  double si = 0.0, sr = 1.0, zr = 1.0, zi = 0.0, h, omdt, hr, hi;

  omdt = 2.0 * M_PI * dt;
  hr = cos(omdt);
  hi = sin(omdt);

  for (i = 0; i < poles; i++) {
    h = zr;
    zr = zr * hr - zi * hi;
    zi = h * hi + zi * hr;
    sr -= coef[i] * zr;
    si -= coef[i] * zi;
  }
  return (sr * sr + si * si);
}

void mem_spec_spectrum(const MemSpecModel *model, unsigned long count,
			double samplingrate, double *freq, double *spec)
{
  unsigned long i;
  double fdt, pow_spec;

  for (i = 0; i < count; i++) {
    fdt = i / (2.0 * (double)count);
    pow_spec = transfer_pow(fdt, model->coef, model->poles);
    freq[i] = fdt * samplingrate;
    spec[i] = model->sigma2 / pow_spec;
  }
}
