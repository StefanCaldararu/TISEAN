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

/* Reentrant core of mutual, factored out of source_c/mutual.c so it has no
   dependency on argv parsing, file-scope globals, or the process-exiting
   error path in the generic rescale_data() library routine it used to
   call. The math here (rescale to [0,1), bin into `partitions` boxes,
   conditional entropy between the series and its own lag-t copy) is
   unchanged from main()/make_cond_entropy(). */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../routines/tsa.h"
#include "../../include/mutual.h"

static double make_cond_entropy(const long *array, unsigned long length,
				 long partitions, long *h1, long *h11,
				 long **h2, long t)
{
  long i, j, hi, hii, count = 0;
  double hpi, hpj, pij, cond_ent = 0.0, norm;

  for (i = 0; i < partitions; i++) {
    h1[i] = h11[i] = 0;
    for (j = 0; j < partitions; j++)
      h2[i][j] = 0;
  }
  for (i = 0; i < (long)length; i++)
    if (i >= t) {
      hii = array[i];
      hi = array[i - t];
      h1[hi]++;
      h11[hii]++;
      h2[hi][hii]++;
      count++;
    }

  norm = 1.0 / (double)count;
  cond_ent = 0.0;

  for (i = 0; i < partitions; i++) {
    hpi = (double)(h1[i]) * norm;
    if (hpi > 0.0) {
      for (j = 0; j < partitions; j++) {
	hpj = (double)(h11[j]) * norm;
	if (hpj > 0.0) {
	  pij = (double)h2[i][j] * norm;
	  if (pij > 0.0)
	    cond_ent += pij * log(pij / hpj / hpi);
	}
      }
    }
  }

  return cond_ent;
}

MutualResult *mutual_compute(const double *series, unsigned long length,
			      long partitions, long corrlength)
{
  unsigned long i;
  long tau;
  double min, interval, x;
  long *array, *h1, *h11, **h2;
  MutualResult *result;

  if (length == 0)
    return NULL;

  min = interval = series[0];
  for (i = 1; i < length; i++) {
    if (series[i] < min)
      min = series[i];
    if (series[i] > interval)
      interval = series[i];
  }
  interval -= min;
  if (interval == 0.0)
    return NULL;

  check_alloc(h1 = (long *)malloc(sizeof(long) * partitions));
  check_alloc(h11 = (long *)malloc(sizeof(long) * partitions));
  check_alloc(h2 = (long **)malloc(sizeof(long *) * partitions));
  for (i = 0; i < (unsigned long)partitions; i++)
    check_alloc(h2[i] = (long *)malloc(sizeof(long) * partitions));
  check_alloc(array = (long *)malloc(sizeof(long) * length));

  for (i = 0; i < length; i++) {
    x = (series[i] - min) / interval;
    if (x < 1.0)
      array[i] = (long)(x * (double)partitions);
    else
      array[i] = partitions - 1;
  }

  if (corrlength >= (long)length)
    corrlength = (long)length - 1;

  check_alloc(result = (MutualResult *)malloc(sizeof(MutualResult)));
  result->length = length;
  result->partitions = partitions;
  result->corrlength = corrlength;
  check_alloc(result->values = (double *)malloc(sizeof(double) * (corrlength + 1)));

  result->values[0] = make_cond_entropy(array, length, partitions, h1, h11, h2, 0);
  for (tau = 1; tau <= corrlength; tau++)
    result->values[tau] = make_cond_entropy(array, length, partitions, h1, h11, h2, tau);

  free(array);
  for (i = 0; i < (unsigned long)partitions; i++)
    free(h2[i]);
  free(h2);
  free(h1);
  free(h11);

  return result;
}

void mutual_free(MutualResult *result)
{
  if (result == NULL)
    return;
  free(result->values);
  free(result);
}
