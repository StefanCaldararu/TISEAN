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

/* Reentrant core of ar-run, factored out of source_c/ar-run.c so it can be
   called both from the ar-run CLI and from other bindings (e.g. Python)
   without going through global state or argv parsing. The math here is
   unchanged from the original AR-model iteration loop. Only the finite
   (explicit -l) case is exposed here: the CLI's -l0/unset "stream forever"
   mode can't be expressed as a bounded return, so it keeps its own loop in
   ar-run.c. */

#ifndef _AR_RUN_H
#define _AR_RUN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Iterates x_n = sum_{j=0}^{poles-1} coeff[j]*x_{n-1-j} + noise, noise ~
   Gaussian(0, var), discarding the first ntrans transient steps, and
   returns a newly allocated array of `length` generated values (free with
   ar_run_free). coeff[0] multiplies the most recent point, coeff[1] the one
   before that, and so on. Returns NULL if poles is 0. */
double *ar_run_generate(unsigned int poles, const double *coeff, double var,
			 unsigned long length, unsigned long ntrans,
			 unsigned long seed);
void ar_run_free(double *series);

#ifdef __cplusplus
}
#endif

#endif
