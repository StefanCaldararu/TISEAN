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

/* Reentrant API for the lyap_k routine: estimates the maximal Lyapunov
   exponent via the method of Kantz. Extracted out of source_c/lyap_k.c so
   it can be called both from the lyap_k CLI and from other bindings
   (e.g. Python) without going through global state, argv parsing, or the
   process-exiting error path in the generic rescale_data() library
   routine it used to call. */

#ifndef _LYAP_K_H
#define _LYAP_K_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  LYAP_K_OK = 0,
  LYAP_K_ERR_ZERO_INTERVAL,     /* series is constant: rescale_data()'s
				    hard-exit case */
  LYAP_K_ERR_TOO_FEW_POINTS     /* maxiter+(maxdim-1)*delay >= length: the
				    CLI's own explicit "Too few points to
				    handle these parameters!" check */
} LyapKError;

typedef struct {
  unsigned int epscount;  /* number of epsilon values actually used (the
			       CLI's -#, possibly forced to 1 when the
			       requested mineps is not smaller than maxeps) */
  unsigned int mindim;      /* clamped mindim actually used (>= 2) */
  unsigned int maxdim;       /* clamped maxdim actually used (>= mindim) */
  unsigned int maxiter;        /* the CLI's -s; every count/lyap row holds
				    maxiter+1 entries for iteration steps
				    0..maxiter */
  double *epsilon;          /* [epscount] neighborhood radius actually used
				 for each row, in the same units as the raw
				 input series (rescaled epsilon multiplied
				 back by the raw data interval) - matches
				 the CLI's "#epsilon=" header and its -V
				 verbosity-level-2 progress line */
  long ***count;              /* [epscount][maxdim-mindim+1][maxiter+1];
				   count[e][d][j] is the number of reference
				   points that contributed to lyap[e][d][j]
				   for embedding dimension mindim+d at
				   epsilon epsilon[e]; 0 means no data for
				   that step, mirroring the CLI's skipping
				   that row entirely in its output */
  double ***lyap;               /* [epscount][maxdim-mindim+1][maxiter+1];
				    raw sum of already-averaged-per-reference-
				    point log divergences; divide by
				    count[e][d][j] to get the CLI's printed
				    value (S(j) = lyap[e][d][j]/count[e][d][j],
				    matching main()'s own formula) - only
				    meaningful where count[e][d][j] > 0 */
} LyapK;

/* Invoked once per epsilon value, after all reference points have been
   processed at that radius (matching the CLI's -V verbosity level 2
   diagnostic line in main()). eps is in the same units as the raw input
   series (the internal [0,1) rescaled epsilon multiplied back by the raw
   data interval), i.e. the same value that ends up in the returned
   result's epsilon[] array. Pass a NULL progress function to
   lyap_k_compute() to skip this reporting entirely. */
typedef void (*LyapKProgressFn)(double eps, void *user_data);

/* Estimates the maximal Lyapunov exponent of series[0..length-1] via the
   method of Kantz, the same way the lyap_k CLI's main() does it: series is
   rescaled to [0,1) on a private copy (the input array is not modified),
   and for each of epscount neighborhood radii (geometrically spaced
   between epsmin and epsmax, in rescaled [0,1) units unless eps0set/
   eps1set is nonzero, in which case epsmin/epsmax are first divided by the
   raw data interval, mirroring the CLI's -r/-R options), a box-assisted
   nearest-neighbor search finds, for the first `reference` points, all
   neighbors within that radius (skipping any within `window` samples of
   the reference point) for every embedding dimension between mindim and
   maxdim, and accumulates the log divergence of the two trajectories for
   maxiter iteration steps ahead.

   reference is clamped like the CLI does: if it exceeds
   length-maxiter-(maxdim-1)*delay (using the caller's un-clamped maxdim,
   matching a CLI quirk - see below), it is silently reduced to that value
   before any other clamping happens.

   mindim/maxdim are clamped exactly like the CLI does, and in the same
   order: the length/maxiter/delay sanity check below runs first against
   the caller's raw mindim/maxdim, and only afterwards are mindim and
   maxdim each floored to 2 and maxdim raised to mindim if it was smaller.
   This means passing e.g. maxdim=1 changes which "too few points" check
   applies compared to passing maxdim=2, exactly as in the original CLI.

   If progress is not NULL, it is called once after each epsilon's
   reference-point loop completes, exactly where the CLI's own progress
   message is printed; pass NULL to skip it.

   delay and window are not validated here: the box-building and
   neighbor-search steps read series[i] for i up to
   length-(maxdim-1)*delay-maxiter, so passing delay large enough that
   (maxdim-1)*delay+maxiter >= length reads out of bounds - this is a
   caller contract (the CLI itself never validates it beyond the "too few
   points" check above), not something checked here. See the Python
   bindings for user-facing validation of that contract.

   Returns NULL and sets *error (if error is not NULL) to:
     - LYAP_K_ERR_ZERO_INTERVAL if series is constant (min == max), or
     - LYAP_K_ERR_TOO_FEW_POINTS if maxiter+(maxdim-1)*delay >= length
       (using the caller's un-clamped maxdim, see above).
   *error is set to LYAP_K_OK on success.

   Caller must free the result with lyap_k_free(). */
LyapK *lyap_k_compute(const double *series, unsigned long length,
		       unsigned int mindim, unsigned int maxdim,
		       unsigned int delay,
		       double epsmin, double epsmax,
		       int eps0set, int eps1set,
		       unsigned int epscount,
		       unsigned long reference,
		       unsigned int maxiter,
		       unsigned int window,
		       LyapKProgressFn progress, void *user_data,
		       LyapKError *error);
void lyap_k_free(LyapK *result);

#ifdef __cplusplus
}
#endif

#endif
