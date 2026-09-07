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

/* Reentrant core of makenoise, factored out of source_c/makenoise.c so it
   has no dependency on argv parsing, file-scope globals, or the
   process-exiting error path in the generic variance() library routine it
   used to call. The math here (equidistri()/gauss()'s noise scaling and
   variance()'s mean/std via a plain sequential sum) is unchanged from
   main(). */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include "../routines/tsa.h"
#include "../../include/makenoise.h"

static void equidistri_add(double *out, unsigned long length, double sigmax,
			    double noiselevel, char absolute)
{
  unsigned long i;
  double limit, equinorm;

  equinorm = (double)ULONG_MAX;
  if (!absolute)
    limit = 2.0 * sqrt(3.0) * sigmax * noiselevel;
  else
    limit = 2.0 * noiselevel;
  for (i = 0; i < length; i++)
    out[i] += (limit * ((double)rnd_1279() / equinorm - 0.5));
}

static void gauss_add(double *out, unsigned long length, double sigmax,
		       double noiselevel, char absolute)
{
  unsigned long i;
  double glevel;

  if (!absolute)
    glevel = noiselevel * sigmax;
  else
    glevel = noiselevel;
  for (i = 0; i < length; i++)
    out[i] += gaussian(glevel);
}

MakeNoise *makenoise_add(double *const *series, unsigned long length,
			  unsigned int dim, double noiselevel, char absolute,
			  char gaussian, unsigned long seed)
{
  unsigned int j;
  unsigned long i;
  double *sigmax;
  MakeNoise *noise;

  if (dim == 0 || length == 0)
    return NULL;

  check_alloc(sigmax = (double *)malloc(sizeof(double) * dim));

  if (!absolute) {
    for (j = 0; j < dim; j++) {
      double av = 0.0, var = 0.0, h;

      for (i = 0; i < length; i++) {
	h = series[j][i];
	av += h;
	var += h * h;
      }
      av /= (double)length;
      var = sqrt(fabs(var / (double)length - av * av));
      if (var == 0.0) {
	free(sigmax);
	return NULL;
      }
      sigmax[j] = var;
    }
  }

  check_alloc(noise = (MakeNoise *)malloc(sizeof(MakeNoise)));
  noise->dim = dim;
  noise->length = length;
  check_alloc(noise->series = (double **)malloc(sizeof(double *) * dim));
  for (j = 0; j < dim; j++) {
    check_alloc(noise->series[j] = (double *)malloc(sizeof(double) * length));
    for (i = 0; i < length; i++)
      noise->series[j][i] = series[j][i];
  }

  rnd_init(seed);
  for (i = 0; i < 10000; i++)
    rnd_1279();

  for (j = 0; j < dim; j++) {
    if (!gaussian)
      equidistri_add(noise->series[j], length, sigmax[j], noiselevel, absolute);
    else
      gauss_add(noise->series[j], length, sigmax[j], noiselevel, absolute);
  }

  free(sigmax);
  return noise;
}

void makenoise_free(MakeNoise *noise)
{
  unsigned int j;

  if (noise == NULL)
    return;
  for (j = 0; j < noise->dim; j++)
    free(noise->series[j]);
  free(noise->series);
  free(noise);
}
