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

/* Reentrant API for the xzero routine: estimates the average cross forecast
   error for a zeroth-order fit between two series given as two columns of
   one file. Extracted out of source_c/xzero.c so it can be called both from
   the xzero CLI and from other bindings (e.g. Python) without going
   through global state, argv parsing, or the process-exiting error paths in
   the generic rescale_data()/variance() library routines it used to call. */

#ifndef _XZERO_H
#define _XZERO_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  unsigned int steps;    /* number of forecast steps, the `step` argument */
  unsigned long clength; /* number of reference points actually scanned,
			     i.e. min(n_ref, length) - steps */
  double *error;         /* [steps] normalized RMS cross-forecast error;
			     error[j] is the error forecasting j+1 steps
			     ahead, normalized by the standard deviation of
			     the (rescaled) series2 */
} XZeroResult;

/* Estimates the average cross forecast error for a zeroth-order fit between
   series1 and series2 (both length `length`), the same way the xzero CLI's
   main() does it: series1 and series2 are each independently rescaled into
   [0,1) on their own copies (neither input array is modified), delay-
   embedded with dimension `dim` and delay `delay`, and for every reference
   point in series2 up to `min(n_ref, length) - step`, a box-assisted
   neighbor search over series1 at a growing radius looks for at least
   `minn` neighbors; once found, the `step`-ahead cross-forecast is a
   zeroth-order (bin average) fit, and the aggregate RMS error for each
   forecast horizon 1..step is normalized by the standard deviation of the
   rescaled series2.

   n_ref mirrors the CLI's -n option (whose ULONG_MAX default sentinel means
   "use `length`"); pass `length` itself for that same default behavior.
   eps0 is the starting neighbor-search radius in rescaled [0,1) units,
   unless epsset is nonzero, in which case eps0 is first divided by the
   average of series1's and series2's own raw data ranges, mirroring the
   CLI's -r option.

   dim, delay, step, minn, eps0, epsf and n_ref are not validated here: the
   CLI itself never produces invalid values for them (its -m/-d/-s/-k/-r/-f/
   -n options all have sane defaults and no explicit validation), so this is
   documented as a caller contract rather than validated here - in
   particular dim must be >= 1 and step must be <= min(n_ref, length), or
   the neighbor search reads out of bounds. See the Python bindings for
   user-facing validation of those two.

   Returns NULL if series1 or series2 is NULL, length == 0, or either
   series is constant (min == max), mirroring rescale_data()'s/variance()'s
   hard-exit cases but without exiting the process.

   Caller must free the result with xzero_free(). */
XZeroResult *xzero_forecast(const double *series1, const double *series2,
			     unsigned long length, unsigned int dim,
			     unsigned int delay, unsigned long n_ref,
			     int minn, double eps0, double epsf,
			     unsigned int step, int epsset);

void xzero_free(XZeroResult *result);

#ifdef __cplusplus
}
#endif

#endif
