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

/* Reentrant API for the lyap_r routine: estimates the maximal Lyapunov
   exponent via the method of Rosenstein et al. Extracted out of
   source_c/lyap_r.c so it can be called both from the lyap_r CLI and from
   other bindings (e.g. Python) without going through global state, argv
   parsing, or the process-exiting error path in the generic
   rescale_data() library routine it used to call. */

#ifndef _LYAP_R_H
#define _LYAP_R_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  unsigned int steps;  /* number of iteration steps, the CLI's -s (default
			   10); found/lyap each hold steps+1 entries for
			   indices 0..steps */
  long *found;          /* [steps+1] number of point pairs that
			    contributed to lyap[i]; 0 means no data for
			    that step, mirroring the CLI's skipping that
			    row entirely in its output */
  double *lyap;          /* [steps+1] raw sum of log(squared divergence)
			     accumulated over the found[i] contributing
			     pairs; divide by found[i] and then by 2.0 to
			     get the CLI's printed value (S(i) =
			     lyap[i]/found[i]/2.0, matching main()'s own
			     formula) - only meaningful where found[i] > 0 */
} LyapR;

/* Invoked once after every growth of the neighbor-search radius eps
   (matching the CLI's -V verbosity level 2 diagnostic line in main()).
   eps is in the same units as the raw input series (the internal [0,1)
   rescaled epsilon multiplied back by the raw data interval); found0 is
   the current number of points that have already found a valid neighbor
   pair for step 0. Pass a NULL progress function to lyap_r_compute() to
   skip this reporting entirely. */
typedef void (*LyapRProgressFn)(double eps, long found0, void *user_data);

/* Estimates the maximal Lyapunov exponent of series[0..length-1] via the
   Rosenstein et al. method, the same way the lyap_r CLI's main() does it:
   series is rescaled to [0,1) on a private copy (the input array is not
   modified), and a box-assisted nearest-neighbor search at a growing
   radius (starting at eps0, in rescaled [0,1) units unless epsset is
   nonzero, in which case eps0 is first divided by the raw data interval,
   mirroring the CLI's -r option) finds, for every reference point, the
   nearest other point more than `mindist` samples away; the log
   divergence of the two trajectories is then accumulated for `steps`
   iteration steps ahead.

   If progress is not NULL, it is called once after each growth of the
   search radius, exactly where the CLI's own progress message is
   printed; pass NULL to skip it.

   dim, delay, steps and mindist are not validated here: the box-building
   step reads series[i] for i up to length-delay*(dim-1)-steps, so passing
   dim, delay or steps large enough that delay*(dim-1)+steps >= length
   reads out of bounds - this is a caller contract (the CLI itself never
   validates it either), not something checked here. dim must also be >=
   1. See the Python bindings for user-facing validation of that
   contract.

   Returns NULL if series is NULL, length == 0, or series is constant
   (min == max), mirroring rescale_data()'s hard-exit case but without
   exiting the process.

   Caller must free the result with lyap_r_free(). */
LyapR *lyap_r_compute(const double *series, unsigned long length,
		       unsigned int dim, unsigned int delay,
		       unsigned int mindist, unsigned int steps,
		       double eps0, int epsset,
		       LyapRProgressFn progress, void *user_data);
void lyap_r_free(LyapR *result);

#ifdef __cplusplus
}
#endif

#endif
