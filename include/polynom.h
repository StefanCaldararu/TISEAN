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

/* Reentrant API for the polynom routine: fits a polynomial of a given order
   to a scalar time series over a delay embedding, via its own internal
   term-encoding scheme (make_coding()/decode() in the original CLI), and
   optionally forecasts it forward. Extracted out of source_c/polynom.c so it
   can be called both from the polynom CLI and from other bindings (e.g.
   Python) without going through global state, argv parsing, or the
   process-exiting error path in the generic variance() library routine it
   used to call. */

#ifndef _POLYNOM_H
#define _POLYNOM_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  POLYNOM_OK = 0,
  POLYNOM_ERR_ZERO_VARIANCE  /* series is constant: the fit's own normalization
				 (dividing the series by its standard deviation)
				 would divide by zero */
} PolynomError;

typedef struct {
  unsigned int dim;      /* embedding dimension, as passed in */
  unsigned int delay;    /* delay, as passed in */
  unsigned int order;    /* polynomial order, as passed in */
  unsigned int plength;  /* number of polynomial terms/coefficients */
  double norm;           /* the series' own standard deviation, used to
			     normalize the fit internally - matches the CLI's
			     "used norm for the fit" output */
  double *coeff;         /* [plength] fitted coefficients, in original
			     (unscaled) data units - already compensated for
			     the internal std-dev normalization, the same way
			     the CLI's printed coefficients are */
  int *exponent;          /* [plength][dim] flattened row-major: exponent
			     [i*dim+j] is the exponent of
			     series[act - j*delay] in term i, decoded the
			     same way the CLI's decode() does it for display */
  double error_insample;  /* in-sample RMS forecast error, normalized by the
			     series' own standard deviation (the fit runs in
			     that normalized space), matching the CLI's
			     "average insample error" */
  int has_outsample;      /* nonzero if error_outsample was computed, i.e. if
			      insample < length was passed in */
  double error_outsample; /* out-of-sample RMS forecast error over
			      [insample, length), same normalization as
			      error_insample; left at 0.0 if
			      has_outsample == 0 */
  unsigned long step;     /* number of forecasted points, 0 if none requested */
  double *forecast;       /* [step] values continuing the series, in original
			      (unscaled) units, iterated forward the same way
			      the CLI's make_cast() does it; NULL if
			      step == 0 */
} PolynomResult;

/* Fits an order-`order` polynomial to series[0..length-1] over a `dim`-
   dimensional, `delay`-delayed embedding, and optionally forecasts it `step`
   points forward, the same way the polynom CLI's main() does it.

   series is the raw (unscaled) input: like the CLI, this function divides by
   the series' own standard deviation internally before fitting, and
   un-scales the fitted coefficients and the forecast back to original data
   units before returning them (this function owns that scaling step itself,
   the same way arima-model's reentrant API owns its own centering). series
   itself is not modified.

   insample selects how much of the series is used to fit the model: if
   insample >= length, all of the series is used for the fit and
   error_outsample is left unset (has_outsample == 0), matching the CLI's -n
   default (unset, i.e. ULONG_MAX). Otherwise the model is fit on
   [0, insample) and error_outsample is the forecast error on the held-out
   [insample, length) range.

   step == 0 skips forecasting (matching the CLI's default when -L is not
   given): forecast is left NULL.

   Returns NULL and sets *error (if error is not NULL) to
   POLYNOM_ERR_ZERO_VARIANCE if series is constant. *error is set to
   POLYNOM_OK on success.

   Neither dim/delay/order/length/insample/step bounds nor the resulting
   normal-equations matrix's singularity are validated here - like the
   original CLI, a singular fit still goes through the shared, process-
   exiting solvele() (see the Python bindings for user-facing validation of
   the input shapes).

   Caller must free the result with polynom_free(). */
PolynomResult *polynom_fit(const double *series, unsigned long length,
			    unsigned int dim, unsigned int delay,
			    unsigned int order, unsigned long insample,
			    unsigned long step, PolynomError *error);
void polynom_free(PolynomResult *result);

#ifdef __cplusplus
}
#endif

#endif
