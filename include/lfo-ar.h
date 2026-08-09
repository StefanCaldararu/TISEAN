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

/* Reentrant API for the lfo-ar routine: estimates the average local-linear
   (AR) forecast error over a range of neighborhood sizes. Extracted out of
   source_c/lfo-ar.c so it can be called both from the lfo-ar CLI and from
   other bindings (e.g. Python) without going through global state, argv
   parsing, or the process-exiting error path in rescale_data(). */

#ifndef _LFO_AR_H
#define _LFO_AR_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  unsigned int dim;
  unsigned long n_rows;  /* number of qualifying neighborhood sizes: only
			     sizes for which more than one scanned point had
			     enough neighbors are kept, exactly like the
			     CLI's own row filter (pfound > 1) */
  double *epsilon;        /* [n_rows], neighborhood size in raw data units */
  double *avg_error;       /* [n_rows], relative forecast error averaged
			       over all dim components */
  double *error;            /* [n_rows][dim] flattened row-major, relative
				forecast error per component */
  double *fraction;          /* [n_rows], fraction of scanned points that
				 had enough neighbors to contribute */
  double *avneighbors;        /* [n_rows], average number of neighbors
				  found per contributing point */
} LfoArResult;

/* Estimates, for a growing sequence of neighborhood sizes, the average
   forecast error of a local-linear (AR) fit, mirroring source_c/lfo-ar.c's
   main()/make_fit(). series is [dim][length] and is not modified: each
   dimension is independently rescaled to [0,1) internally the same way
   rescale_data() does it, but without exiting the process on a constant
   dimension - if any dimension is constant (max == min), NULL is returned
   and, if bad_value is non-NULL, *bad_value is set to that dimension's
   constant value (matching what rescale_data()'s "data ranges from %e to
   %e" message would have printed - both %e are that same value).

   embed/delay are the CLI's -m (embedding dimension part)/-d. step is the
   CLI's -s (forecast horizon). causal is the CLI's -C (causality window
   half-width); pass step to match the CLI's default (unset -C, i.e.
   causal = step). iterations is the CLI's -i (raw/unclamped: pass length
   to match the CLI's default of "whole series").

   eps0/eps1 are the neighborhood size range to scan, matching the CLI's
   -r/-R. If eps0_raw/eps1_raw is non-zero, the respective value is
   interpreted in the original (raw) data units and divided by the largest
   of the per-dimension raw intervals before use, matching the CLI's -r/-R
   flags; if zero, the value is used as-is in the already-rescaled [0,1)
   space (the CLI's own defaults, 1.e-3/1.0, are already in that space).
   epsf is the CLI's -f growth factor for the neighborhood size.

   Returns NULL (without touching *bad_value) if dim == 0, embed == 0,
   length == 0, step < 0, step >= length, or iterations < step. */
LfoArResult *lfo_ar_compute(double *const *series, unsigned long length,
			     unsigned int dim, unsigned int embed,
			     unsigned int delay, int step,
			     unsigned long causal, unsigned long iterations,
			     double eps0, char eps0_raw,
			     double eps1, char eps1_raw, double epsf,
			     double *bad_value);
void lfo_ar_free(LfoArResult *result);

#ifdef __cplusplus
}
#endif

#endif
