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

/* Reentrant API for the rbf routine: fits a radial-basis-function model to
   a scalar time series and can forecast it forward. Extracted out of
   source_c/rbf.c so it can be called both from the rbf CLI and from other
   bindings (e.g. Python) without going through global state, argv parsing,
   or the process-exiting error paths in the generic variance()/
   rescale_data()/solvele() library routines it used to call. */

#ifndef _RBF_H
#define _RBF_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  RBF_OK = 0,
  RBF_ERR_ZERO_VARIANCE,   /* series is constant: rescale_data()'s/
			       variance()'s normalization would divide by
			       zero */
  RBF_ERR_SINGULAR_MATRIX  /* the least-squares normal-equations matrix for
			       the RBF weights is singular */
} RBFError;

typedef struct {
  unsigned int dim;
  unsigned int delay;
  unsigned int centers;       /* number of RBF centers actually used, i.e.
				  min(requested centers, length) */
  unsigned long step;         /* forecast horizon the model was fit for */
  unsigned long insample;     /* number of points actually used to fit the
				  model, i.e. min(requested insample,
				  length) */
  unsigned long length;
  double **center;             /* [centers][dim], center coordinates in
				   original (unscaled) data units */
  double variance;              /* RBF kernel width parameter, in original
				    (unscaled) data units */
  double *coefs;                 /* [centers+1]; coefs[0] is the fit
				     intercept in original units, coefs[1..
				     centers] are the per-center weights
				     scaled to original units */
  double insample_error;         /* normalized in-sample RMS forecast
				     error (dimensionless) */
  int has_outsample_error;       /* nonzero if insample < length */
  double outsample_error;        /* normalized out-of-sample RMS forecast
				     error on [insample, length); 0.0 if
				     has_outsample_error == 0 */
  unsigned long cast_length;
  double *cast;                   /* [cast_length] forecasted values in
				      original units, continuing the series;
				      NULL if cast_length == 0 */
} RBFResult;

/* Fits an RBF model to series[0..length-1] and forecasts it cast_length
   points forward, the same way the rbf CLI's main() does it. series is not
   modified.

   centers is clamped to length (matching the CLI's -p handling); the
   number of centers actually used is returned in the result as `centers`.
   drift is a boolean: nonzero applies the same 20-iteration repulsion
   optimization to the initial center placement as the CLI's default
   behaviour (drift == 0 matches the CLI's -X).

   This is a caller contract, not something validated in this reentrant
   core (see the Python bindings for user-facing validation of it):
     - length must be > (dim - 1) * delay, for the center-placement and
       forecasting windows to stay in bounds.
     - min(centers, length) must be >= 2: the center-spacing computation
       divides by (that value - 1).
     - min(insample, length) must be >= step: the model-fitting and
       in-sample-error loops subtract step from it before comparing
       against an unsigned loop index.

   Returns NULL and sets *error (if error is not NULL) to:
     - RBF_ERR_ZERO_VARIANCE if series is constant, or
     - RBF_ERR_SINGULAR_MATRIX if the normal-equations matrix for the RBF
       weights is singular.
   *error is set to RBF_OK on success.

   Caller must free the result with rbf_free(). */
RBFResult *rbf_fit(const double *series, unsigned long length,
		    unsigned int dim, unsigned int delay, unsigned int centers,
		    int drift, unsigned long step, unsigned long insample,
		    unsigned long cast_length, RBFError *error);
void rbf_free(RBFResult *result);

#ifdef __cplusplus
}
#endif

#endif
