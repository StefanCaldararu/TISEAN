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

/* Reentrant API for the lyap_spec routine: estimates the spectrum of
   Lyapunov exponents using the method of Sano and Sawada. Extracted out of
   source_c/lyap_spec.c so it can be called both from the lyap_spec CLI and
   from other bindings (e.g. Python) without going through global state,
   argv parsing, or the process-exiting error paths in the generic
   rescale_data() library routine it used to call. */

#ifndef _LYAP_SPEC_H
#define _LYAP_SPEC_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  unsigned int dimension;  /* CLI's -m first number: # of components */
  unsigned int embed;      /* CLI's -m second number: embedding dimension
			       per component */
  unsigned int alldim;     /* dimension*embed, the number of Lyapunov
			       exponents */
  unsigned long count;     /* number of iterations actually performed
			       (min(iterations, length-1)) */
  double *exponents;       /* [alldim] final average Lyapunov exponents
			       (natural-log units per iteration step), i.e.
			       the CLI's last printed data row divided by
			       count */
  double *rel_forecast_error; /* [dimension] average relative one-step-
				  ahead forecast error per component,
				  matching the CLI's "Average relative
				  forecast errors" line */
  double *abs_forecast_error; /* [dimension] average absolute one-step-
				  ahead forecast error per component,
				  matching the CLI's "Average absolute
				  forecast errors" line */
  double avg_neighborhood_size; /* matches the CLI's "Average Neighborhood
				    Size" */
  double avg_num_neighbors;     /* matches the CLI's "Average num. of
				    neighbors" */
  double ky_dimension;          /* estimated Kaplan-Yorke dimension,
				    matches the CLI's "estimated KY-Dimension"
				 */
} LyapSpec;

/* Invoked once per iteration, right after gram_schmidt() re-orthonormalizes
   the tangent vectors for that step - the same point in the computation
   where the CLI's own wall-clock-gated progress line would be printed.
   count is the 1-based iteration counter; exponents (length alldim) is the
   running average up to and including this iteration, i.e. what the CLI
   would print already divided by count; is_last is nonzero on the final
   iteration (the CLI always prints that row regardless of wall-clock
   timing, so callers that want to mirror the CLI's periodic-plus-final
   output should apply their own timing gate but always act on is_last).
   Pass a NULL progress function to lyap_spec_compute() to skip this
   reporting entirely. */
typedef void (*LyapSpecProgressFn)(unsigned long count, const double *exponents,
				    unsigned int alldim, int is_last,
				    void *user_data);

/* Estimates the spectrum of Lyapunov exponents of series (a [dimension]
   array of [length]-element rows), the same way the lyap_spec CLI's main()
   does it: series is not modified - an internal copy is rescaled to [0,1)
   per component and, if inverse is nonzero, time-reversed (matching the
   CLI's -I option). A box-assisted nearest-neighbor search with a growing
   radius (starting at epsmin/EPSSTEP and capped at 1.0; in rescaled [0,1)
   units unless epsset is nonzero, in which case epsmin is first divided by
   the largest per-dimension raw data range, mirroring the CLI's -r option)
   fits a local-linear model at each of up to `iterations` points and
   iterates the tangent space forward via Gram-Schmidt reorthonormalization,
   matching the CLI's -m/-r/-f/-k/-n/-I options. The delay between embedding
   coordinates is fixed at 1, matching the CLI (its -d option is disabled in
   the CLI's own show_options()/scan_options()).

   Returns NULL if:
     - minneighbors > length - (embed-1) - 1, i.e. the series is too short
       to ever find that many neighbors, mirroring the CLI's own pre-flight
       check (exit code 51 there)
     - any dimension's raw data is constant (min == max), mirroring
       rescale_data()'s hard-exit case (exit code 11 there) but without
       exiting the process
     - not enough neighbors are found for some reference point even after
       growing the search radius up to 1.0, mirroring the CLI's own "Not
       enough neighbors found" hard-exit case (exit code 50 there)

   dimension and embed must both be >= 1; that is a caller contract
   mirroring the CLI's own lack of validation, not something checked here
   (see the Python bindings for user-facing validation of that contract).

   Caller must free the result with lyap_spec_free(). */
LyapSpec *lyap_spec_compute(double *const *series, unsigned long length,
			     unsigned int dimension, unsigned int embed,
			     unsigned long iterations,
			     double epsmin, int epsset, double epsstep,
			     unsigned int minneighbors, int inverse,
			     LyapSpecProgressFn progress, void *user_data);
void lyap_spec_free(LyapSpec *result);

#ifdef __cplusplus
}
#endif

#endif
