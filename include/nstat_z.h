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

/* Reentrant API for the nstat_z routine: tests for nonstationarity by
   splitting a (univariate) time series into `pieces` equal-length segments
   and, for each selected pair of segments (first, second), fitting a
   zeroth-order (local-constant) forecaster on `first` and measuring its
   rms forecast error when applied to reference points drawn from `second`,
   scaled by `second`'s own rms. Extracted out of source_c/nstat_z.c so it
   can be called both from the nstat_z CLI and from other bindings (e.g.
   Python) without going through global state, argv parsing, or the
   process-exiting error paths in the generic variance()/rescale_data()
   library routines it used to call. */

#ifndef _NSTAT_Z_H
#define _NSTAT_Z_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  NSTAT_Z_OK = 0,
  NSTAT_Z_ERR_ZERO_INTERVAL,    /* the series' raw data range (max - min) is
				    zero */
  NSTAT_Z_ERR_ZERO_VARIANCE,    /* some piece's data, after the series is
				    rescaled to [0,1), has zero variance
				    (only possible via rounding - an
				    exact-zero-interval series is always
				    caught by NSTAT_Z_ERR_ZERO_INTERVAL
				    first) */
  NSTAT_Z_ERR_TOO_MANY_PIECES   /* `pieces` is so large that a piece has
				    fewer than `minn` usable reference
				    points */
} NstatZError;

typedef struct {
  unsigned int pieces;   /* number of pieces the series was split into (the
			     CLI's -#) */
  unsigned long n_pairs; /* number of (first, second) piece pairs actually
			     computed, i.e. how many entries `first`,
			     `second` and `value` hold */
  unsigned int *first;   /* [n_pairs]: 0-indexed piece used to fit the
			     zeroth-order model (the CLI's first printed
			     column, 1-indexed) */
  unsigned int *second;  /* [n_pairs]: 0-indexed piece the fit is tested
			     against (the CLI's second printed column,
			     1-indexed) */
  double *value;         /* [n_pairs]: rms forecast error of the `first`
			     piece's fit on `second`'s reference points,
			     divided by `second`'s own rms - exactly the
			     value the CLI prints as its third column */
} NstatZ;

/* series is [0..length-1] and is not modified: it is internally rescaled
   to its own [0,1) range on a private copy, the same way the nstat_z CLI's
   main() does it (rescale_data()), but without exiting the process on a
   degenerate (constant) series.

   pieces is the CLI's -# (required, must be >= 1): the series is split
   into `pieces` equal-length, non-overlapping segments of
   (length - (dim-1)*delay) / pieces points each.

   first_window/second_window are `pieces`-length arrays of 0/1 flags
   selecting which pieces are candidates for the "first" (fitted) and
   "second" (tested-against) role, matching the CLI's -1/-2 options with a
   plain (non "+offset") argument; NULL means "all pieces" (the CLI's
   default when -1/-2 is not given).

   first_offset/second_offset are the CLI's -1/-2 options given in "+N"
   form (a window of pieces within N of the other index), or -1 if that
   form was not used (the CLI's default). When first_offset/second_offset
   is not -1, it takes precedence over first_window/second_window for
   selecting the corresponding role, exactly like the CLI's main().

   dim/delay are the CLI's -m/-d. minn is the CLI's -k (minimum number of
   neighbors required before a fit is accepted). step is the CLI's -s
   (forecast horizon). causal is the CLI's -C (causality window
   half-width) - callers must resolve its "default: steps" themselves
   (i.e. pass `step` if the CLI's -C was not given), matching what the
   CLI's own main() does before calling this function.

   center/centerset are the CLI's -n (number of reference points in the
   window) and whether it was given; if centerset is 0, center is ignored
   and every point of a piece is used as a reference point (the CLI's
   "default: all"), after which the actual number of reference points used
   is clamped to fit within a piece given `dim`/`delay`/`step` - exactly
   the CLI's own clamp.

   eps0/epsf are the CLI's -r (starting neighborhood size)/-f (growth
   factor). If epsset is non-zero, eps0 is interpreted in the original
   (raw) data units and divided by the series' raw interval before use,
   matching the CLI's -r flag; if zero, eps0 is used as-is in the
   already-rescaled [0,1) space (the CLI's own default, 1.e-3, is already
   in that space).

   Returns NULL and sets *error (if error is not NULL) to:
     - NSTAT_Z_ERR_ZERO_INTERVAL if the series' raw data range is zero,
     - NSTAT_Z_ERR_ZERO_VARIANCE if some piece's rescaled data has zero
       variance, or
     - NSTAT_Z_ERR_TOO_MANY_PIECES if `pieces` leaves fewer than `minn`
       usable reference points per piece.
   *error is set to NSTAT_Z_OK on success.

   Like the CLI, this does not bound how long the internal neighbor search
   can run: if minn can never be satisfied for some reference point (e.g.
   minn is close to or exceeds a piece's length), the search loop does not
   terminate. Callers must ensure minn is achievable, same as the CLI's
   own (undocumented) requirement.

   Caller must free the result with nstat_z_free(). */
NstatZ *nstat_z_compute(const double *series, unsigned long length,
			 unsigned int pieces,
			 const char *first_window, const char *second_window,
			 int first_offset, int second_offset,
			 unsigned int dim, unsigned int delay,
			 unsigned int minn, unsigned long step,
			 unsigned long causal, unsigned long center,
			 char centerset, double eps0, char epsset,
			 double epsf, NstatZError *error);
void nstat_z_free(NstatZ *result);

#ifdef __cplusplus
}
#endif

#endif
