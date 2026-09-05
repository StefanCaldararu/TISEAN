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

/* Reentrant API for the ghkss routine: multivariate noise reduction via the
   GHKSS algorithm. Extracted out of source_c/ghkss.c so it can be called
   both from the ghkss CLI and from other bindings (e.g. Python) without
   going through global state, argv parsing, or the process-exiting error
   paths in the generic rescale_data()/eigen() library routines it used to
   call. The math (per-component rescale, the box-assisted neighbor search,
   the local correction-matrix eigendecomposition, and the trend removal) is
   unchanged from main()/mmb()/fmn()/make_correction()/handle_trend()/
   set_correction(). */

#ifndef _GHKSS_H
#define _GHKSS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  double epsilon;   /* neighborhood size at this step, in original (raw)
			data units */
  unsigned long count; /* cumulative number of points corrected (in the
			   correction pass) or trend-subtracted (in the
			   trend pass) once this epsilon was reached */
} GHKSSEpsStep;

typedef struct {
  double *shift; /* [comp]: average shift applied to each component this
		     iteration, in original (raw) data units */
  double *rms;   /* [comp]: average rms size of the correction applied to
		     each component this iteration, in original (raw) data
		     units */

  GHKSSEpsStep *correction_steps; /* neighborhood sizes tried while
				      searching for enough neighbors to
				      correct every point, one entry per
				      epsilon value scanned this iteration */
  unsigned long n_correction_steps;

  GHKSSEpsStep *trend_steps; /* neighborhood sizes used while evaluating and
				 removing the trend, one entry per epsilon
				 value from correction_steps (same count) */
  unsigned long n_trend_steps;

  char mineps_reset; /* non-zero if the minimal neighborhood size was halved
			 (divided by sqrt(2)) for the next iteration, i.e.
			 some point already had enough neighbors at the very
			 first (smallest) epsilon tried this iteration */
  double mineps_after; /* the minimal neighborhood size to use for the next
			   iteration, in original (raw) data units - equal
			   to the value this iteration started with unless
			   mineps_reset is non-zero */

  double **series; /* [comp][length]: the series after this iteration's
		       correction, in original (raw) data units */
} GHKSSIteration;

typedef struct {
  unsigned int comp;
  unsigned long length;
  unsigned int iterations;
  GHKSSIteration *iters; /* [iterations] */
} GHKSSResult;

typedef enum {
  GHKSS_ERR_TOO_MANY_NEIGHBORS = 1, /* length < minn: can never find minn
					neighbors */
  GHKSS_ERR_ZERO_INTERVAL,          /* some component of series is constant
					(mirrors rescale_data()'s exit path);
					*bad_value is set to that component's
					constant value */
  GHKSS_ERR_EIGEN_NO_CONVERGE       /* the eigenvalue solver failed to
					converge for some point's local
					correction matrix (mirrors eigen()'s
					exit path) */
} GHKSSError;

/* Performs GHKSS multivariate noise reduction on series ([comp][length]),
   matching source_c/ghkss.c's main() loop over `iterations` passes. series
   is not modified; each component is independently rescaled to [0,1)
   internally the same way rescale_data() does it.

   embed/delay/qdim/minn/euclidean are the CLI's -m (embedding dim part)/
   -d/-q/-k/-2. If eps_set is non-zero, mineps is interpreted in the
   original (raw) data units and divided by the largest per-component raw
   interval before use, matching the CLI's -r; if zero, mineps is ignored
   and the default of 1/1000 (in the internally-rescaled [0,1) space) is
   used instead, matching the CLI's own default.

   Returns NULL and sets *error (and, for GHKSS_ERR_ZERO_INTERVAL,
   *bad_value) on failure instead of exiting the process:
   - GHKSS_ERR_TOO_MANY_NEIGHBORS if length < minn
   - GHKSS_ERR_ZERO_INTERVAL if some component of series is constant
   - GHKSS_ERR_EIGEN_NO_CONVERGE if the eigendecomposition failed to
     converge for some point's local correction matrix */
GHKSSResult *ghkss_reduce(double *const *series, unsigned long length, unsigned int comp,
			   unsigned int embed, unsigned int delay, unsigned int qdim,
			   unsigned int minn, double mineps, char eps_set,
			   unsigned int iterations, char euclidean,
			   GHKSSError *error, double *bad_value);
void ghkss_free(GHKSSResult *result);

#ifdef __cplusplus
}
#endif

#endif
