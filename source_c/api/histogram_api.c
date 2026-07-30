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

/* Reentrant core of histogram, factored out of source_c/histogram.c so it
   has no dependency on argv parsing, file-scope globals, or the
   process-exiting error paths in the generic variance()/rescale_data()
   library routines it used to call. The math here (mean/std via a plain
   sequential sum, min/max scan, rescale to [0,1), clamp the top edge into
   the last bin, floor(x*base) per point) is unchanged from main(). */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../routines/tsa.h"
#include "../../include/histogram.h"

Histogram *histogram_compute(const double *series, unsigned long length,
			      unsigned long base)
{
  unsigned long i;
  long j;
  double h, average, var, min, interval, size, size2, x;
  double *rescaled;
  Histogram *hist;

  if (length == 0)
    return NULL;

  average = var = 0.0;
  for (i = 0; i < length; i++) {
    h = series[i];
    average += h;
    var += h * h;
  }
  average /= (double)length;
  var = sqrt(fabs(var / (double)length - average * average));

  min = interval = series[0];
  for (i = 1; i < length; i++) {
    if (series[i] < min)
      min = series[i];
    if (series[i] > interval)
      interval = series[i];
  }
  interval -= min;

  /* Constant data (interval == 0) always also means var == 0 and vice
     versa, so this one check stands in for both of the original's
     variance()/rescale_data() exit paths. */
  if (interval == 0.0)
    return NULL;

  check_alloc(hist = (Histogram *)malloc(sizeof(Histogram)));
  hist->base = base;
  hist->min = min;
  hist->interval = interval;
  hist->average = average;
  hist->var = var;
  hist->box = NULL;

  if (base == 0)
    return hist;

  check_alloc(rescaled = (double *)malloc(sizeof(double) * length));
  for (i = 0; i < length; i++)
    rescaled[i] = (series[i] - min) / interval;

  check_alloc(hist->box = (long *)malloc(sizeof(long) * base));
  for (i = 0; i < base; i++)
    hist->box[i] = 0;

  size = 1. / (double)base;
  size2 = size / 2.0;
  for (i = 0; i < length; i++) {
    x = rescaled[i];
    if (x > (1.0 - size2))
      x = 1.0 - size2;
    j = (long)(x * (double)base);
    hist->box[j]++;
  }

  free(rescaled);
  return hist;
}

void histogram_free(Histogram *h)
{
  if (h == NULL)
    return;
  free(h->box);
  free(h);
}
