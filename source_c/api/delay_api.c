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

/* Reentrant core of delay, factored out of source_c/delay.c's stdout
   output path so it has no dependency on argv parsing or file-scope
   globals. The indexing math here (rundel incrementing continuously
   across every column of a row) is unchanged from the original stdout
   loop in main(). */

#include <stdio.h>
#include <stdlib.h>
#include "../routines/tsa.h"
#include "../../include/delay.h"

DelayResult *delay_compute(double *const *series, unsigned long length,
			    unsigned int indim, const unsigned int *format,
			    const unsigned int *delays)
{
  unsigned int alldim, i, k;
  unsigned long maxdelay, t;
  DelayResult *d;

  if (indim == 0)
    return NULL;

  alldim = 0;
  for (i = 0; i < indim; i++)
    alldim += format[i];
  if (alldim == 0)
    return NULL;

  maxdelay = 0;
  for (k = 0; k < alldim; k++)
    if (delays[k] > maxdelay)
      maxdelay = delays[k];

  check_alloc(d = (DelayResult *)malloc(sizeof(DelayResult)));
  d->alldim = alldim;
  d->vectors = NULL;
  d->n_vectors = (length > maxdelay) ? (length - maxdelay) : 0;

  if (d->n_vectors == 0)
    return d;

  check_alloc(d->vectors = (double *)malloc(sizeof(double) * d->n_vectors * alldim));

  for (t = 0; t < d->n_vectors; t++) {
    unsigned long time_index = t + maxdelay;
    unsigned int rundel = 0, col, out = 0;

    for (col = 0; col < indim; col++) {
      unsigned int emb;
      for (emb = 0; emb < format[col]; emb++)
	d->vectors[t * alldim + out++] = series[col][time_index - delays[rundel++]];
    }
  }

  return d;
}

void delay_free(DelayResult *d)
{
  if (d == NULL)
    return;
  free(d->vectors);
  free(d);
}
