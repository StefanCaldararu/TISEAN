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

/* Reentrant API for the histogram routine: bins data rescaled to [0,1)
   into `base` equal-width intervals. Extracted out of source_c/histogram.c
   so it can be called both from the histogram CLI and from other bindings
   (e.g. Python) without going through global state, argv parsing, or the
   process-exiting error paths in variance()/rescale_data(). */

#ifndef _HISTOGRAM_H
#define _HISTOGRAM_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  unsigned long base;  /* number of bins */
  double min;           /* minimum of the raw (un-rescaled) series */
  double interval;      /* max-min of the raw series */
  double average;       /* mean of the raw series */
  double var;            /* standard deviation of the raw series */
  long *box;            /* [base] counts per bin, NULL if base == 0 */
} Histogram;

/* Bins series[0..length-1] into `base` equal-width intervals over its own
   [min,max] range, the same way the CLI does it (rescale to [0,1), clamp
   the top edge into the last bin, floor(x*base) per point). series is not
   modified. Returns NULL if length == 0 or the data is constant (min ==
   max, mirroring rescale_data()'s/variance()'s hard-exit case but without
   exiting the process). base == 0 is not an error: it just yields an empty
   histogram (box == NULL) with min/interval/average/var still filled in. */
Histogram *histogram_compute(const double *series, unsigned long length,
			      unsigned long base);
void histogram_free(Histogram *h);

#ifdef __cplusplus
}
#endif

#endif
