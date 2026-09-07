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

/* Reentrant API for the arima-model routine: differences and fits a
   multivariate AR (or, if a combined AR/I/MA order is requested, an
   iteratively-refined ARIMA) model and can iterate it forward. Extracted
   out of source_c/arima-model.c so it can be called both from the
   arima-model CLI and from other bindings (e.g. Python) without going
   through global state, argv parsing, or the process-exiting error path in
   the generic variance() library routine it used to call. */

#ifndef _ARIMA_MODEL_H
#define _ARIMA_MODEL_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  ARIMA_MODEL_OK = 0,
  ARIMA_MODEL_ERR_TOO_MANY_POLES,  /* poles is 0 or >= length (after the I-th
				       differencing), or the ARMA refinement was
				       requested (arpoles+ipoles+mapoles > 0) and
				       arpoles or mapoles is >= that length */
  ARIMA_MODEL_ERR_ZERO_VARIANCE    /* some component's data (after the I-th
				       differencing) has zero variance, so it
				       can't be centered */
} ARIMAModelError;

typedef struct {
  unsigned int dim;
  unsigned long length;      /* series length after the I-th differencing */
  unsigned int poles;        /* initial AR-fit order, as requested */
  unsigned int arpoles;      /* AR order of the combined ARMA fit, as
				 requested (0 if the ARMA refinement did not
				 run) */
  unsigned int ipoles;       /* differencing order actually applied */
  unsigned int mapoles;      /* MA order of the combined ARMA fit, as
				 requested (0 if the ARMA refinement did not
				 run) */
  unsigned char arimaset;    /* whether the ARMA refinement ran, i.e.
				 arpoles+ipoles+mapoles > 0 */
  unsigned int order;        /* poles actually used to index coeff/residuals:
				 equals poles if !arimaset, else
				 max(arpoles, mapoles) */
  unsigned int size;         /* number of coefficients per dim row:
				 dim*poles if !arimaset, else
				 dim*(arpoles+mapoles) */
  double *average;           /* [dim], per-component mean subtracted while
				 centering the (differenced) series */
  double **series;           /* [dim][length], the differenced, centered
				 series that was actually fit */
  double **coeff;            /* [dim][size] */
  double *rms_error;         /* [dim], RMS one-step forecast error per
				 component */
  double **residuals;        /* [dim][length]; entries [0..order-1] are
				 zero, entries [order..length-1] are
				 one-step-ahead prediction residuals */
  unsigned int **aindex;     /* [2][size]; aindex[0][i]/aindex[1][i] give the
				 (component, lag) coeff[.][i] belongs to.
				 component < dim means the series itself;
				 dim <= component < 2*dim (arimaset only)
				 means the residual/MA term */
  unsigned int realiter;     /* number of ARMA refinement iterations actually
				 run (0 if !arimaset) */
  double **xdiff;            /* [realiter][dim]; per-iteration RMS change of
				 the residuals, NULL if !arimaset */
  double *diffcoeff;         /* [realiter]; per-iteration RMS change of the
				 coefficients (see arima_model_api.c for the
				 exact - dim-only, not size-only - comparison
				 this mirrors from the original), NULL if
				 !arimaset */
} ARIMAModel;

/* series is [dim][length] and is not modified - differencing and centering
   are both done internally on a private working copy, mirroring the
   arima-model CLI's own make_difference()/set_averages_to_zero() calls.

   poles is the initial AR-fit order (-p); arpoles/ipoles/mapoles are the
   combined AR/I/MA fit order (-P; pass 0,0,0 to skip the ARMA refinement and
   just return the initial AR fit); iterations is the max number of ARMA
   refinement iterations (-I); convergence is the convergence threshold for
   that refinement (-e).

   Returns NULL and sets *error (if error is not NULL) to
   ARIMA_MODEL_ERR_TOO_MANY_POLES if poles is 0 or >= the series length after
   differencing, or if the ARMA refinement was requested and arpoles or
   mapoles is >= that length; or to ARIMA_MODEL_ERR_ZERO_VARIANCE if some
   component's differenced data has zero variance. *error is set to
   ARIMA_MODEL_OK on success. */
ARIMAModel *arima_model_fit(double *const *series, unsigned long length,
			     unsigned int dim, unsigned int poles,
			     unsigned int arpoles, unsigned int ipoles,
			     unsigned int mapoles, unsigned int iterations,
			     double convergence, ARIMAModelError *error);
void arima_model_free(ARIMAModel *model);

/* Iterates the fitted model forward for ilength steps, seeding the
   bootstrap resampler (rand_arb_dist(), drawing from the fitted residuals)
   and the Gaussian/uniform generator with seed. Returns a newly allocated
   [ilength][dim] array (free with arima_model_iterate_free). */
double **arima_model_iterate(const ARIMAModel *model, unsigned long ilength,
			      unsigned long seed);
void arima_model_iterate_free(double **iterated, unsigned long ilength);

#ifdef __cplusplus
}
#endif

#endif
