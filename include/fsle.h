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

/* Reentrant API for the fsle routine: estimates the finite-size Lyapunov
   exponent spectrum via the method of Vulpiani et al. Extracted out of
   source_c/fsle.c so it can be called both from the fsle CLI and from other
   bindings (e.g. Python) without going through global state, argv parsing,
   or the process-exiting error paths in the generic variance()/
   rescale_data() library routines it used to call. */

#ifndef _FSLE_H
#define _FSLE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  FSLE_OK = 0,
  FSLE_ERR_ZERO_VARIANCE,  /* series is constant, checked both on the raw
			       series and, redundantly like the original,
			       again after the internal [0,1] rescale */
  FSLE_ERR_ZERO_INTERVAL,  /* series' raw range (max - min) is zero: the
			       internal rescale to [0,1] would divide by
			       zero */
  FSLE_ERR_EPS_TOO_LARGE   /* the requested/derived starting epsilon is not
			       smaller than the maximal epsilon (the data's
			       own scale) */
} FSLEError;

typedef struct {
  unsigned long n;    /* number of populated (eps, lyapunov, count) entries:
			  only the exponentially-spaced epsilon bins that saw
			  at least one divergence event, the same filter
			  fsle.c's main() applies (factor > 0.0) before
			  printing a row */
  double *eps;          /* [n] epsilon of each bin, in the original series'
			    units (scaled back up by the series' own raw
			    max-min range) */
  double *lyapunov;      /* [n] finite-size Lyapunov exponent estimate for
			    that bin: accumulated log-divergence divided by
			    accumulated time */
  long *count;            /* [n] number of divergence events contributing to
			    that bin */
} FSLEResult;

/* Estimates the finite-size Lyapunov exponent spectrum of series[0..length-1]
   the same way the fsle CLI's main() does it: internally centers/rescales a
   private copy of series to [0,1] (series itself is not modified), then
   tracks pairs of nearby trajectory points as they diverge across
   exponentially-spaced (factor sqrt(2)) epsilon bins starting from eps0.

   dim/delay form the embedding (window width delay*(dim-1)); mindist is the
   minimum index separation between two points before they're considered a
   valid neighbor pair.

   eps0 is the starting epsilon. If epsset is 0, eps0 is treated as a
   fraction of the (rescaled) series' standard deviation (the CLI's default
   when -r is not given: eps0 defaults to 1e-3 that way). If epsset is
   non-zero, eps0 is used as an absolute epsilon in the series' own raw
   units (the CLI's -r value).

   dim, delay and mindist are not validated here: the box-building step
   reads series[i] for i up to length-delay*(dim-1)-1-mindist, so passing
   values large enough that delay*(dim-1)+1+mindist >= length reads out of
   bounds - this is a caller contract (the CLI itself never validates it
   either), not something checked here. dim must also be >= 1. See the
   Python bindings for user-facing validation of that contract.

   Returns NULL and sets *error (if error is not NULL) to:
     - FSLE_ERR_ZERO_VARIANCE if series is constant, or
     - FSLE_ERR_ZERO_INTERVAL if series' raw max-min range is zero, or
     - FSLE_ERR_EPS_TOO_LARGE if the resulting starting epsilon is not
       smaller than the data's own maximal epsilon.
   *error is set to FSLE_OK on success.

   Caller must free the result with fsle_free(). */
FSLEResult *fsle_compute(const double *series, unsigned long length,
			  unsigned int dim, unsigned int delay,
			  unsigned int mindist, double eps0, int epsset,
			  FSLEError *error);
void fsle_free(FSLEResult *result);

#ifdef __cplusplus
}
#endif

#endif
