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

/* Reentrant API for the polynomp routine: fits a multivariate polynomial
   to a scalar time series by least squares and forecasts it forward.
   Extracted out of source_c/polynomp.c so it can be called both from the
   polynomp CLI and from other bindings (e.g. Python) without going through
   global state, argv parsing, or the process-exiting error paths in the
   generic variance()/solvele() library routines it used to call. */

#ifndef _POLYNOMP_H
#define _POLYNOMP_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  POLYNOMP_OK = 0,
  POLYNOMP_ERR_ZERO_VARIANCE,   /* series is constant: the FCE normalization
				    (dividing by the series' standard
				    deviation) would divide by zero */
  POLYNOMP_ERR_SINGULAR_MATRIX  /* the least-squares normal-equations matrix
				    is singular; no unique fit exists for the
				    given data/terms */
} PolynompError;

typedef struct {
  unsigned int dim;      /* embedding dimension, as passed in */
  unsigned int delay;    /* delay, as passed in */
  unsigned int plength;  /* number of polynomial terms/coefficients */
  double *param;         /* [plength] fitted coefficients, in the same
			     term order as the input `order` matrix */
  double fce_insample;   /* in-sample forecast error, RMS residual over the
			     fitted range divided by the series' standard
			     deviation */
  int has_outsample;     /* nonzero if fce_outsample was computed, i.e. if
			     insample < length was passed in */
  double fce_outsample;  /* out-of-sample forecast error, same normalization
			     as fce_insample, over [insample+1, length); left
			     at 0.0 if has_outsample == 0 */
  unsigned long step;    /* number of forecasted points */
  double *forecast;      /* [step] values continuing the series, iterated
			     forward from its own tail the same way the CLI's
			     make_cast() does it */
} PolynompResult;

/* Fits a degree-mixed polynomial to series[0..length-1] and forecasts it
   `step` points forward, the same way the polynomp CLI's main() does it.
   series is not modified.

   order is a flattened [plength][dim] row-major matrix of non-negative
   exponents (the same layout PolyParResult.params uses, e.g. as produced by
   polypar_generate()): term i of the polynomial is
     product over j in [0,dim) of series[act - j*delay] ^ order[i*dim+j]
   plength must be >= 1 and dim must be >= 1: like the original recursion
   polypar_generate() is based on, dim == 0 is not a supported input (the
   fit's window width is computed as (dim - 1) * delay, computed as
   unsigned, so it underflows). Similarly plength == 0 is not supported: the
   normal-equations solve indexes down from plength - 1, which underflows
   for plength == 0. Neither is checked here - this is a caller contract,
   not something validated in this reentrant core (see the Python bindings
   for user-facing validation of it).

   insample selects how much of the series is used to fit the model: if
   insample >= length, all of the series is used for the fit and
   fce_outsample is left unset (has_outsample == 0), matching the CLI's -n
   default (unset, i.e. ULONG_MAX). Otherwise the model is fit on
   [0, insample) and fce_outsample is the forecast error on the held-out
   [insample+1, length) range.

   Forecasting past the end of the series (make_cast) reads the last
   (dim - 1) * delay + 1 points of series; length must be greater than
   (dim - 1) * delay for this to be a valid range - like the CLI, this is
   not validated here either.

   Returns NULL and sets *error (if error is not NULL) to:
     - POLYNOMP_ERR_ZERO_VARIANCE if series is constant, or
     - POLYNOMP_ERR_SINGULAR_MATRIX if the normal-equations matrix built
       from `order`/insample/series is singular.
   *error is set to POLYNOMP_OK on success.

   Caller must free the result with polynomp_free(). */
PolynompResult *polynomp_fit(const double *series, unsigned long length,
			      const unsigned int *order, unsigned int plength,
			      unsigned int dim, unsigned int delay,
			      unsigned long insample, unsigned long step,
			      PolynompError *error);
void polynomp_free(PolynompResult *result);

#ifdef __cplusplus
}
#endif

#endif
