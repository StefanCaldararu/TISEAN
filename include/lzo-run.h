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

/* Reentrant API for the lzo-run routine: iterates a local zeroth-order
   (nearest-neighbor) forecast forward for multivariate data. Extracted out
   of source_c/lzo-run.c so it can be called both from the lzo-run CLI and
   from other bindings (e.g. Python) without going through global state,
   argv parsing, or the process-exiting error paths in the generic
   variance()/rescale_data() library routines it used to call. */

#ifndef _LZO_RUN_H
#define _LZO_RUN_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  LZO_RUN_OK = 0,
  LZO_RUN_ERR_ZERO_INTERVAL,  /* a component's raw data range (max - min) is
				  zero */
  LZO_RUN_ERR_ZERO_VARIANCE   /* a component's data, after being rescaled to
				  [0,1), has zero variance (only possible via
				  rounding - an exact-zero-interval component
				  is always caught by LZO_RUN_ERR_ZERO_INTERVAL
				  first) */
} LzoRunError;

typedef struct {
  unsigned int dim;
  unsigned long length;  /* == flength: number of iterated forecast points */
  double *series;         /* [length][dim] row-major, the iterated forecast
			      trajectory, scaled back into the original (raw)
			      units of the input series */
} LzoRun;

/* series is [dim][length] and is not modified: each component is internally
   rescaled to its own [0,1) range the same way the lzo-run CLI's main()
   does it (rescale_data() followed by variance(), the latter only used to
   size the optional Gaussian noise), but on a private copy and without
   exiting the process on a degenerate component.

   embed/delay are the CLI's -m (embedding dimension part)/-d. minn is the
   CLI's -k (minimum number of neighbors required before a forecast point is
   accepted).

   fix_neighbors is the CLI's -K ("fix # of neighbors"): when non-zero, once
   at least minn neighbors are found the search radius keeps growing to
   collect the full box population, the minn closest are kept via a partial
   selection sort, and the search radius accumulated across iterations
   (mirroring the CLI's setsort/epsilon0/count bookkeeping) seeds the next
   iteration's starting radius. NOTE: the CLI's own global for this
   (setsort) defaults to 1 and scan_options() only ever sets it to 1 too
   (there is no code path that sets it back to 0), so the shipped CLI always
   calls this function with fix_neighbors != 0 regardless of whether -K was
   given - that quirk is preserved here rather than "fixed", since this is a
   pure refactor of the existing behavior.

   flength is the CLI's -L (number of iterations/forecast points to
   produce).

   eps0/epsf are the CLI's -r (starting neighborhood size)/-f (growth
   factor). If epsset is non-zero, eps0 is interpreted in the original (raw)
   data units and divided by the largest of the per-component raw intervals
   before use, matching the CLI's -r flag; if zero, eps0 is used as-is in
   the already-rescaled [0,1) space (the CLI's own default, 1.e-3, is
   already in that space).

   noise_pct/setnoise are the CLI's -% (percentage of each component's
   rescaled variance added as Gaussian noise to every forecast point) and
   whether it was given at all with a positive value (the CLI only enables
   noise when -% is explicitly passed and the resulting value is > 0.0,
   regardless of the flag's own default of 10.0 - so "not given" and
   "given as 0 or negative" both mean no noise, matching scan_options()).

   seed seeds the Gaussian noise generator via rnd_init() (only consulted
   when setnoise is non-zero); rnd_init() is a process-wide, one-shot seed
   (see source_c/routines/rand.c), so it only has an effect the first time
   any routine using it is called in a process.

   Returns NULL and sets *error (if error is not NULL) to:
     - LZO_RUN_ERR_ZERO_INTERVAL if some component's raw data range is zero,
       or
     - LZO_RUN_ERR_ZERO_VARIANCE if some component's rescaled data has zero
       variance.
   *error is set to LZO_RUN_OK on success.

   Like the CLI, this does not bound how long the internal neighbor search
   can run: if minn can never be satisfied for some forecast step (e.g. minn
   is close to or exceeds length), the search loop does not terminate.
   Callers must ensure minn is achievable, same as the CLI's own
   (undocumented) requirement. Callers must also ensure dim >= 1, embed >= 1
   and length > (embed-1)*delay, matching the CLI's own implicit
   requirements.

   Caller must free the result with lzo_run_free(). */
LzoRun *lzo_run_forecast(double *const *series, unsigned long length,
			   unsigned int dim, unsigned int embed,
			   unsigned int delay, unsigned int minn,
			   char fix_neighbors, unsigned long flength,
			   double eps0, char epsset, double epsf,
			   double noise_pct, char setnoise,
			   unsigned long seed, LzoRunError *error);
void lzo_run_free(LzoRun *result);

#ifdef __cplusplus
}
#endif

#endif
