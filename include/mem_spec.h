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

/* Reentrant API for the mem_spec routine: fits an AR power-spectrum model
   via Burg's method and evaluates the implied power spectrum. Extracted
   out of source_c/mem_spec.c (getcoefs()/powcoef()) so it can be called
   both from the mem_spec CLI and from other bindings (e.g. Python)
   without going through global state, argv parsing, or the
   process-exiting error path in variance(). */

#ifndef _MEM_SPEC_H
#define _MEM_SPEC_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  unsigned long poles;  /* number of AR poles fitted */
  double sigma2;         /* residual variance of the Burg fit ("pm") */
  double *coef;          /* [poles] AR coefficients */
} MemSpecModel;

/* Fits an AR model to series[0..length-1] via Burg's method, the same way
   the mem_spec CLI does it (mean-center the raw series, then run the
   Burg recursion for `poles` reflection coefficients). series is not
   modified. Returns NULL if poles >= length (too many poles for the
   data) or if series is constant (zero variance, mirroring variance()'s
   hard-exit case but without exiting the process). */
MemSpecModel *mem_spec_fit(const double *series, unsigned long length,
			    unsigned long poles);
void mem_spec_free(MemSpecModel *model);

/* Evaluates the power spectrum implied by `model` at `count` frequencies
   fdt = i/(2*count) for i in 0..count-1 (i.e. covering [0,0.5) in units
   of the sampling rate), matching powcoef()'s transfer-function
   evaluation in mem_spec.c. freq and spec must each hold `count` doubles
   allocated by the caller; freq[i] is fdt*samplingrate and spec[i] is
   model->sigma2 divided by the squared transfer function magnitude
   (unscaled - the mem_spec CLI additionally divides by sqrt(length)
   when writing to a file but not when writing to stdout; callers that
   need that scaling must apply it themselves). */
void mem_spec_spectrum(const MemSpecModel *model, unsigned long count,
			double samplingrate, double *freq, double *spec);

#ifdef __cplusplus
}
#endif

#endif
