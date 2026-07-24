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

/* Reentrant API for the ar-model routine: fits a multivariate AR model
   and can iterate it forward. Extracted out of source_c/ar-model.c so it
   can be called both from the ar-model CLI and from other bindings
   (e.g. Python) without going through global state or argv parsing. */

#ifndef _AR_MODEL_H
#define _AR_MODEL_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  unsigned int dim;
  unsigned int poles;
  unsigned long length;
  double **coeff;      /* [dim][dim*poles], AR coefficients */
  double *rms_error;   /* [dim], RMS one-step forecast error per component */
  double **residuals;  /* [dim][length]; entries [poles..length-1] are the
			   one-step-ahead prediction residuals, matching the
			   input series' own centering/scaling */
} ARModel;

/* series is [dim][length] and is expected to already be centered (e.g. via
   variance()/mean subtraction) the same way the ar-model CLI does it.
   Returns NULL if poles is 0 or poles >= length. */
ARModel *ar_model_fit(double *const *series, unsigned long length,
		       unsigned int dim, unsigned int poles);
void ar_model_free(ARModel *model);

/* Iterates the fitted model forward for ilength steps, seeding the
   Gaussian noise generator with seed. Returns a newly allocated
   [ilength][dim] array (free with ar_model_iterate_free). */
double **ar_model_iterate(const ARModel *model, unsigned long ilength,
			   unsigned long seed);
void ar_model_iterate_free(double **iterated, unsigned long ilength);

#ifdef __cplusplus
}
#endif

#endif
