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

/* Reentrant API for the lfo-run routine: makes a local (linear or zeroth
   order) fit for multivariate data and iterates a forecast trajectory
   forward. Extracted out of source_c/lfo-run.c so it can be called both
   from the lfo-run CLI and from other bindings (e.g. Python) without going
   through global state, argv parsing, or the process-exiting error path in
   the generic rescale_data() library routine it used to call. */

#ifndef _LFO_RUN_H
#define _LFO_RUN_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  LFO_RUN_OK = 0,
  LFO_RUN_ERR_ZERO_INTERVAL,    /* a component's raw data range (max - min)
				    is zero */
  LFO_RUN_ERR_ESCAPED_REGION    /* the iterated forecast left the rescaled
				    [-1,2) data region partway through; the
				    result still holds every point computed
				    up to and including the failing one,
				    mirroring what the CLI already printed
				    before it used to exit() */
} LfoRunError;

typedef struct {
  unsigned int dim;
  unsigned long length;  /* number of forecast points actually produced:
			     equals flength on success, or fewer if
			     LFO_RUN_ERR_ESCAPED_REGION cut the run short */
  double *series;         /* [length][dim] row-major, the iterated forecast
			      trajectory, scaled back into the original (raw)
			      units of the input series */
} LfoRun;

/* series is [dim][length] and is not modified: each component is internally
   rescaled to its own [0,1) range the same way the lfo-run CLI's main()
   does it (rescale_data()), but on a private copy and without exiting the
   process on a degenerate component.

   embed/delay are the CLI's -m (embedding dimension part)/-d. minn is the
   CLI's -k (minimum number of neighbors required before a forecast point is
   accepted). do_zeroth is the CLI's -0 (use a zeroth order fit instead of
   the local-linear fit).

   flength is the CLI's -L (number of iterations/forecast points to
   produce).

   eps0/epsf are the CLI's -r (starting neighborhood size)/-f (growth
   factor). If epsset is non-zero, eps0 is interpreted in the original (raw)
   data units and divided by the largest of the per-component raw intervals
   before use, matching the CLI's -r flag; if zero, eps0 is used as-is in
   the already-rescaled [0,1) space (the CLI's own default, 1.e-3, is
   already in that space).

   Returns NULL and sets *error (if error is not NULL) to
   LFO_RUN_ERR_ZERO_INTERVAL if some component's raw data range is zero.
   *error is set to LFO_RUN_OK on success.

   If the iterated forecast leaves the rescaled [-1,2) data region (the same
   condition the CLI used to detect with "newcast[j] > 2.0 || newcast[j] <
   -1.0"), a non-NULL result is still returned holding every point computed
   up to and including the one that escaped, and *error is set to
   LFO_RUN_ERR_ESCAPED_REGION.

   Like the CLI, this does not bound how long the internal neighbor search
   can run: if minn can never be satisfied for some forecast step (e.g. minn
   is close to or exceeds length), the search loop does not terminate.
   Callers must ensure minn is achievable, same as the CLI's own
   (undocumented) requirement. Callers must also ensure dim >= 1, embed >= 1
   and length > (embed-1)*delay, matching the CLI's own implicit
   requirements.

   Caller must free the result with lfo_run_free(). */
LfoRun *lfo_run_forecast(double *const *series, unsigned long length,
			  unsigned int dim, unsigned int embed,
			  unsigned int delay, unsigned int minn,
			  char do_zeroth, unsigned long flength,
			  double eps0, char epsset, double epsf,
			  LfoRunError *error);
void lfo_run_free(LfoRun *result);

#ifdef __cplusplus
}
#endif

#endif
