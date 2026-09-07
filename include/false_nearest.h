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

/* Reentrant API for the false_nearest routine: estimates the fraction of
   false nearest neighbors as a function of embedding dimension. Extracted
   out of source_c/false_nearest.c so it can be called both from the
   false_nearest CLI and from other bindings (e.g. Python) without going
   through global state, argv parsing, or the process-exiting error paths
   in the generic variance()/rescale_data() library routines it used to
   call. */

#ifndef _FALSE_NEAREST_H
#define _FALSE_NEAREST_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  FALSE_NEAREST_OK = 0,
  FALSE_NEAREST_ERR_ZERO_INTERVAL,     /* a component's raw data range
					   (max - min) is zero */
  FALSE_NEAREST_ERR_ZERO_VARIANCE,     /* a component's data, after being
					   rescaled to [0,1), has zero
					   variance (only possible via
					   rounding - an exact-zero-interval
					   component is always caught by
					   FALSE_NEAREST_ERR_ZERO_INTERVAL
					   first) */
  FALSE_NEAREST_ERR_NOT_ENOUGH_POINTS  /* for some embedding dimension, no
					   neighbor pair was found before
					   epsilon grew past the data's own
					   scale */
} FalseNearestError;

typedef struct {
  unsigned long n;      /* number of embedding dimensions computed, i.e.
			    max(0, maxemb - minemb + 1) */
  unsigned int *dimension; /* [n] total embedding dimension (comp * emb for
			       the emb-th row), matching the first column
			       the CLI prints */
  double *fraction;      /* [n] fraction of false nearest neighbors */
  double *avg_eps;       /* [n] average neighbor distance at which a false
			     neighbor was decided, scaled back into the
			     original series' units */
  double *sigma_eps;     /* [n] standard deviation of that distance, scaled
			     back into the original series' units */
} FalseNearest;

/* Computes the fraction of false nearest neighbors of series[0..comp-1]
   (each of length `length`) for every total embedding dimension
   comp*emb, emb running from minemb to maxemb inclusive, the same way the
   false_nearest CLI's main() does it: internally rescales a private copy
   of each component to [0,1) and requires the shared (minimum across
   components) variance of that rescaled data to build a box-assisted
   nearest-neighbor search, starting from epsilon `eps0` and growing it by
   sqrt(2) until enough neighbor pairs are found or the search exceeds the
   data's own scale (governed by `rt`, the escape factor). `theiler` is the
   Theiler window: candidate neighbors within `theiler` samples of a point
   (in time) are excluded. `delay` is the reconstruction delay used when
   comp > 1.

   series is not modified. If minemb > maxemb, returns a result with n == 0
   (no error) - the loop simply doesn't run, matching the CLI.

   The box-assisted search reads series[c][i] for i up to
   length - (maxemb+1)*delay - 1, so callers must ensure
   length > (maxemb+1)*delay whenever minemb <= maxemb (the CLI itself
   never validates this either); minemb must also be >= 1, since dim is
   computed as emb*comp-1 in an unsigned type. These are caller contracts,
   not checked here - see the Python bindings for user-facing validation of
   them.

   Returns NULL and sets *error (if error is not NULL) to:
     - FALSE_NEAREST_ERR_ZERO_INTERVAL if some component's raw data range
       is zero, or
     - FALSE_NEAREST_ERR_ZERO_VARIANCE if some component's rescaled data
       has zero variance, or
     - FALSE_NEAREST_ERR_NOT_ENOUGH_POINTS if no neighbor pair was found
       for some embedding dimension.
   *error is set to FALSE_NEAREST_OK on success.

   Caller must free the result with false_nearest_free(). */
FalseNearest *false_nearest_compute(double *const *series, unsigned long length,
				     unsigned int comp, unsigned int delay,
				     unsigned int minemb, unsigned int maxemb,
				     unsigned long theiler, double rt,
				     double eps0, FalseNearestError *error);
void false_nearest_free(FalseNearest *fn);

#ifdef __cplusplus
}
#endif

#endif
