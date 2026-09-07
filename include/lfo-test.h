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

/* Reentrant API for the lfo-test routine: estimates the average forecast
   error of a local-linear fit at a single (growing) neighborhood size per
   point. Extracted out of source_c/lfo-test.c so it can be called both from
   the lfo-test CLI and from other bindings (e.g. Python) without going
   through global state, argv parsing, or the process-exiting error path in
   the generic rescale_data() library routine it used to call. */

#ifndef _LFO_TEST_H
#define _LFO_TEST_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  unsigned int comp;
  unsigned long length;    /* same as the input series' length */
  double *rms_error;        /* [comp], relative forecast error per
				 component: sqrt(error/norm)/rms, where rms is
				 the standard deviation of that component
				 after rescaling to [0,1) */
  double *individual;         /* [comp][length] flattened row-major, in the
				  original (raw) units of the input series.
				  Entry [c][i] holds (forecast-actual)*interval
				  for every scanned point i that found more
				  than `minn` neighbors; every other entry
				  (points outside the scanned range, or that
				  never found enough neighbors) is 0.0 */
} LfoTest;

/* Estimates the average forecast error of a local-linear fit, mirroring
   source_c/lfo-test.c's main()/make_fit(). series is [comp][length] and is
   not modified: each component is independently rescaled to [0,1) internally
   the same way rescale_data() does it, but on a private copy and without
   exiting the process on a constant component - if any component is
   constant (max == min), NULL is returned and, if bad_value is non-NULL,
   *bad_value is set to that component's constant value (matching what
   rescale_data()'s "data ranges from %e to %e" message would have printed -
   both %e are that same value).

   embed/delay are the CLI's -m (embedding dimension part)/-d. minn is the
   CLI's -k (minimum number of neighbors required before a point's forecast
   is accepted). step is the CLI's -s (forecast horizon). causal is the
   CLI's -C (causality window half-width); pass step to match the CLI's
   default (unset -C, i.e. causal = step). iterations is the CLI's -n
   (raw/unclamped: pass length to match the CLI's default of "whole
   series").

   eps0 is the neighborhood size to start with, matching the CLI's -r. If
   epsset is non-zero, eps0 is interpreted in the original (raw) data units
   and divided by the largest of the per-component raw intervals before use,
   matching the CLI's -r flag; if zero, eps0 is used as-is in the
   already-rescaled [0,1) space (the CLI's own default, 1.e-3, is already in
   that space). epsf is the CLI's -f growth factor for the neighborhood
   size.

   Returns NULL (without touching *bad_value) if comp == 0, embed == 0,
   length == 0, or length - (embed-1)*delay < minn (the data set is too
   short to find enough neighbors for the fit).

   Like the CLI, this does not bound how long the search for a growing
   neighborhood size can run: if minn can never be satisfied for some point
   (e.g. minn is close to or exceeds length), the outer loop does not
   terminate. Callers must ensure minn is achievable, same as the CLI's own
   (undocumented) requirement.

   Caller must free the result with lfo_test_free(). */
LfoTest *lfo_test_forecast(double *const *series, unsigned long length,
			    unsigned int comp, unsigned int embed,
			    unsigned int delay, unsigned int minn,
			    unsigned int step, unsigned long causal,
			    unsigned long iterations,
			    double eps0, char epsset, double epsf,
			    double *bad_value);
void lfo_test_free(LfoTest *result);

#ifdef __cplusplus
}
#endif

#endif
