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

/* Reentrant API for the lzo-test routine: estimates the average forecast
   error for a zeroth-order (local-constant) fit from a multidimensional
   time series, for a range of forecast horizons. Extracted out of
   source_c/lzo-test.c so it can be called both from the lzo-test CLI and
   from other bindings (e.g. Python) without going through global state,
   argv parsing, or the process-exiting error paths in the generic
   variance()/rescale_data() library routines it used to call. */

#ifndef _LZO_TEST_H
#define _LZO_TEST_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  LZO_TEST_OK = 0,
  LZO_TEST_ERR_ZERO_INTERVAL,  /* a component's raw data range (max - min)
				   is zero */
  LZO_TEST_ERR_ZERO_VARIANCE   /* a component's data, after being rescaled
				   to [0,1), has zero variance (only possible
				   via rounding - an exact-zero-interval
				   component is always caught by
				   LZO_TEST_ERR_ZERO_INTERVAL first) */
} LzoTestError;

typedef struct {
  unsigned int dim;      /* number of components */
  unsigned long step;    /* number of forecast horizons computed (the
			     CLI's -s) */
  double *error;          /* [step][dim] row-major: relative forecast error
			      (rms forecast error over rms of the component,
			      scaled) for horizon i+1 (row i), component j -
			      exactly the values the CLI's first output block
			      prints */
  unsigned long n_ref;      /* number of reference points covered by
				`diffs` below; 0 if none */
  double *diffs;              /* [n_ref][dim] row-major: per-reference-point
				  one-step forecast differences, scaled back
				  into the original series' units - only
				  printed by the CLI when verbosity &
				  VER_USR1 (its -V2) */
} LzoTest;

/* series is [dim][length] and is not modified: each component is
   internally rescaled to its own [0,1) range and centered/scaled the same
   way the lzo-test CLI's main() does it (rescale_data() followed by
   variance()), but on a private copy and without exiting the process on a
   degenerate component.

   embed/delay are the CLI's -m (embedding dimension part)/-d. minn is the
   CLI's -k (minimum number of neighbors required before a fit is
   accepted). step is the CLI's -s (number of forecast horizons). refstep
   is the CLI's -S (temporal distance between reference points). causal is
   the CLI's -C (causality window half-width) - callers must resolve its
   "default: steps" themselves (i.e. pass `step` if the CLI's -C was not
   given), matching what the CLI's own main() does before calling this
   function.

   clength/clengthset are the CLI's -n (number of reference points) and
   whether it was given; if clengthset is 0, clength is ignored and the
   full series is used instead (matching the CLI's "default: length"),
   after which the actual number of reference points scanned is clamped
   to fit within `length` given `step` and `refstep` - exactly the CLI's
   own clamp.

   eps0/epsf are the CLI's -r (starting neighborhood size)/-f (growth
   factor). If epsset is non-zero, eps0 is interpreted in the original
   (raw) data units and divided by the average of the per-component raw
   intervals before use, matching the CLI's -r flag; if zero, eps0 is used
   as-is in the already-rescaled [0,1) space (the CLI's own default,
   1.e-3, is already in that space).

   Returns NULL and sets *error (if error is not NULL) to:
     - LZO_TEST_ERR_ZERO_INTERVAL if some component's raw data range is
       zero, or
     - LZO_TEST_ERR_ZERO_VARIANCE if some component's rescaled data has
       zero variance.
   *error is set to LZO_TEST_OK on success.

   Like the CLI, this does not bound how long the internal neighbor search
   can run: if minn can never be satisfied for some reference point (e.g.
   minn is close to or exceeds length), the search loop does not
   terminate. Callers must ensure minn is achievable, same as the CLI's
   own (undocumented) requirement.

   Caller must free the result with lzo_test_free(). */
LzoTest *lzo_test_compute(double *const *series, unsigned long length,
			   unsigned int dim, unsigned int embed,
			   unsigned int delay, unsigned int minn,
			   unsigned long step, unsigned long refstep,
			   unsigned long causal, unsigned long clength,
			   char clengthset, double eps0, char epsset,
			   double epsf, LzoTestError *error);
void lzo_test_free(LzoTest *result);

#ifdef __cplusplus
}
#endif

#endif
