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

/* Reentrant API for the delay routine: builds delay-embedding vectors out
   of one or more input columns. Extracted out of source_c/delay.c so it
   can be called both from the delay CLI and from other bindings (e.g.
   Python) without going through global state or argv parsing. Only the
   stdout output path's math is extracted (see delay.c's main(): its -o
   file-output path builds the same kind of vectors via a slightly
   different, pre-existing indexing loop and is left untouched by the
   refactor so the CLI's -o behavior is provably unchanged). */

#ifndef _DELAY_H
#define _DELAY_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  unsigned int alldim;      /* sum(format[i]) for i in [0, indim) */
  unsigned long n_vectors;  /* number of output delay vectors */
  double *vectors;          /* [n_vectors * alldim], row-major, NULL if n_vectors == 0 */
} DelayResult;

/* Builds delay vectors out of `indim` input columns (series[0..indim-1],
   each `length` points long), the same way delay.c's main() does for its
   stdout output path.

   format[i] (i in [0,indim)) is the number of embedded coordinates taken
   from column i; every format[i] must be >= 1 (a column contributing zero
   embedded coordinates is not a supported input and will overrun
   `delays`/the output row size - callers must filter it out before
   calling this function).

   delays is a flat array of alldim = sum(format) cumulative time-lags (in
   samples), laid out column-major within each column in the same order
   as format: the first format[0] entries are column 0's per-coordinate
   lags, the next format[1] entries are column 1's, and so on. The first
   lag of every column (delays[0], and the first entry of every column's
   block) must be 0.

   For output row t (t = 0..n_vectors-1, corresponding to time index
   t+maxdelay in the input series, maxdelay = max(delays)), the k-th
   coordinate of the row (k = 0..alldim-1, using the same flat layout as
   format/delays) is series[col(k)][t + maxdelay - delays[k]], where
   col(k) is whichever column produced index k.

   Returns NULL if indim == 0 or alldim (sum of format) == 0. If `length`
   is not larger than maxdelay, the result is valid but empty (n_vectors
   == 0, vectors == NULL) - matching the CLI, which prints nothing rather
   than erroring in that case. */
DelayResult *delay_compute(double *const *series, unsigned long length,
			    unsigned int indim, const unsigned int *format,
			    const unsigned int *delays);
void delay_free(DelayResult *d);

#ifdef __cplusplus
}
#endif

#endif
