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

/* Reentrant API for the polyback routine: fits a polynomial to a scalar
   time series and backward-eliminates terms one at a time, tracking the
   in-/out-of-sample forecast error at each reduced term count. Extracted
   out of source_c/polyback.c so it can be called both from the polyback
   CLI and from other bindings (e.g. Python) without going through global
   state, argv parsing, or the process-exiting error paths in the generic
   variance()/solvele() library routines it used to call. */

#ifndef _POLYBACK_H
#define _POLYBACK_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  POLYBACK_OK = 0,
  POLYBACK_ERR_ZERO_VARIANCE,    /* series is constant: the error
				     normalization (dividing by the series'
				     standard deviation) would divide by
				     zero */
  POLYBACK_ERR_SINGULAR_MATRIX   /* the least-squares normal-equations
				     matrix built for the full model, or for
				     some candidate reduced model during
				     backward elimination, is singular */
} PolybackError;

typedef struct {
  unsigned int dim;             /* embedding dimension, as passed in */
  unsigned int delay;           /* delay, as passed in */
  unsigned long n_terms;        /* initial number of polynomial terms
				    (order's row count), as passed in */
  double error_in;              /* in-sample forecast error of the full
				    (n_terms-term) model, normalized by the
				    series' standard deviation */
  int has_outsample;            /* nonzero if error_out was computed, i.e.
				    if insample < length was passed in */
  double error_out;             /* out-of-sample forecast error of the full
				    model, same normalization as error_in,
				    over [insample+1, length); left at 0.0 if
				    has_outsample == 0 */
  unsigned int n_levels;        /* number of backward-elimination steps
				    performed: n_terms - down_to (down_to
				    clamped into [1, n_terms] first, same as
				    the CLI) */
  unsigned int *level_n_terms;  /* [n_levels] terms remaining after each
				    step, descending from n_terms-1 to the
				    clamped down_to */
  double *level_error_in;       /* [n_levels] in-sample forecast error at
				    each step, same normalization as
				    error_in */
  double *level_error_out;      /* [n_levels] out-of-sample forecast error
				    at each step, same normalization as
				    error_in; 0.0 for every entry if
				    has_outsample == 0 */
  unsigned long *removed_index; /* [n_levels] 0-based row index into the
				    input `order` array of the term
				    eliminated at that step */
} PolybackResult;

/* Fits a polynomial to series[0..length-1] and repeatedly eliminates the
   term whose removal least hurts the forecast error (out-of-sample if
   available, else in-sample), one term at a time, the same way the
   polyback CLI's main() does it. series is not modified.

   order is a flattened [n_terms][dim] row-major matrix of non-negative
   exponents (the same layout PolyParResult.params/polynomp_fit's `order`
   use): term i of the polynomial is the product over j in [0, dim) of
   series[act - j*delay] ^ order[i*dim + j]. n_terms must be >= 1 and dim
   must be >= 1; neither is checked here - this is a caller contract, not
   something validated in this reentrant core (see the Python bindings for
   user-facing validation of it).

   insample selects how much of the series is used to fit the model: if
   insample >= length, all of the series is used for the fit and error_out/
   level_error_out are left at 0.0 (has_outsample == 0), matching the CLI's
   -n default (unset, i.e. ULONG_MAX). Otherwise the model is fit on
   [0, insample) and the *_out fields are the forecast error on the held-out
   [insample+1, length) range.

   step is the forecast horizon used in the fit criterion (the CLI's -s;
   unrelated to any output length). down_to is the term count to reduce
   down to (the CLI's -#); it is clamped to 1 if it is 0 or greater than
   n_terms, exactly like the CLI does.

   Returns NULL and sets *error (if error is not NULL) to:
     - POLYBACK_ERR_ZERO_VARIANCE if series is constant, or
     - POLYBACK_ERR_SINGULAR_MATRIX if a normal-equations matrix built
       during the full fit or during backward elimination is singular.
   *error is set to POLYBACK_OK on success.

   Caller must free the result with polyback_free(). */
PolybackResult *polyback_fit(const double *series, unsigned long length,
			      const unsigned int *order, unsigned long n_terms,
			      unsigned int dim, unsigned int delay,
			      unsigned long insample, unsigned int step,
			      unsigned int down_to, PolybackError *error);
void polyback_free(PolybackResult *result);

#ifdef __cplusplus
}
#endif

#endif
