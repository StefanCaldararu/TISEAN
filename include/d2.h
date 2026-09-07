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

/* Reentrant API for the d2 routine: estimates the correlation sum,
   -dimension and -entropy of a (possibly delay-embedded, multivariate)
   series via box-assisted nearest-neighbour search. Extracted out of
   source_c/d2.c so it can be called both from the d2 CLI and from other
   bindings (e.g. Python) without going through global state, argv parsing,
   or the process-exiting error paths in the generic rescale_data() library
   routine (and d2.c's own "delay vector too large" check) it used to call. */

#ifndef _D2_H
#define _D2_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  D2_OK = 0,
  D2_ERR_VECTOR_TOO_LARGE_FOR_LENGTH, /* length - (embed-1)*delay <= 0: the
					  delay vector would be longer than
					  the whole series */
  D2_ERR_RESCALE_ZERO_INTERVAL        /* rescale was requested and some
					  component's raw data range
					  (max-min) is zero */
} D2Error;

typedef struct {
  unsigned int dim;       /* number of series components used */
  unsigned int embed;     /* max embedding dimension */
  unsigned int n_blocks;  /* dim*embed, number of per-block output rows in
			      c2/h2/d2 below - block 0 is the unembedded
			      (1D) case, matching the CLI's "#dim=" blocks */
  unsigned int howoften;  /* number of epsilon scales computed */
  double *eps;            /* [howoften], epsilon in original (unrescaled,
			      unless rescale was requested) data units -
			      shared by every row of c2/h2/d2 below, matching
			      the CLI's column 1 in all three output files */
  double **c2;             /* [n_blocks][howoften], correlation integral
			       (.c2 file); NaN where the CLI's own norm[j]>0
			       guard would have skipped printing the row */
  double **h2;             /* [n_blocks][howoften], correlation entropy
			       (.h2 file); NaN where the CLI's own guard
			       (found[.][j]>0, see d2.c) would have skipped
			       printing the row */
  double **d2;             /* [n_blocks][howoften], local slope / D2
			       estimate (.d2 file); column 0 is always NaN
			       (the CLI's own loop starts at column index 1),
			       and other entries are NaN where the CLI's own
			       guard would have skipped printing the row */
} D2Result;

/* Invoked once per center point processed by the main search loop - the
   same point in the computation where the CLI's own wall-clock-gated
   (.stat/.c2/.h2/.d2) dump would run. snapshot reflects the correlation
   sum/entropy/slope tables as accumulated up to and including
   centers_treated points (same shape and NaN-gating as the value
   d2_compute() itself returns - it is NOT the same object, and is only
   valid for the duration of the call). current_eps_max is the current
   (possibly narrowed) maximum epsilon, matching the CLI's own .stat
   "Maximal epsilon in the moment" line, except clamped to the last valid
   epsilon bin instead of reproducing the 1-past-the-end array read the
   CLI's own epsm[imin] would perform in the rare case every epsilon bin
   has saturated MAXFOUND (a latent bug in the original, not behavior worth
   replicating in library code). is_last is nonzero on the final
   invocation (either the search ran through every center point, or every
   epsilon bin saturated MAXFOUND) - the CLI always dumps on that
   invocation regardless of wall-clock timing, so callers that want to
   mirror the CLI's periodic-plus-final output should apply their own
   timing gate but always act on is_last. Pass a NULL progress function to
   d2_compute() to skip this reporting (and the snapshot-building work it
   requires) entirely. */
typedef void (*D2ProgressFn)(const D2Result *snapshot,
			      unsigned long centers_treated,
			      double current_eps_max, int is_last,
			      void *user_data);

/* series is [dim][length], not modified (a rescaled working copy is made
   internally when rescale is set, mirroring the CLI's own in-place
   rescale_data() calls but without mutating the caller's array).

   embed is the max embedding dimension (the CLI's -M dim,embed second
   value); delay is the reconstruction delay (-d); theiler_window is the
   Theiler window (-t); howoften is the number of epsilon scales (-#);
   maxfound is the max number of neighbour pairs per epsilon before that
   epsilon is dropped from further center points (-N; 0 means unlimited,
   matching the CLI's -N 0); rescale requests rescaling each component to
   its own [0,1) range before processing (-E).

   eps_max/eps_min are the requested epsilon bounds (-R/-r). If
   eps_max_absolute/eps_min_absolute is zero, eps_max/eps_min are
   interpreted as fractions of the data's max range across components (the
   CLI's default: eps_max=1.0, eps_min=1e-3, meaning "whole range" and
   "range/1000"); if nonzero, eps_max/eps_min are used directly as absolute
   epsilons in data units (the CLI's -R/-r with an explicit value).

   Caller contracts (not checked here - see the Python bindings for
   user-facing validation, matching false_nearest/boxcount's convention):
   dim >= 1, embed >= 1, delay >= 1, howoften >= 1.

   Returns NULL and sets *error (if error is not NULL) to
   D2_ERR_VECTOR_TOO_LARGE_FOR_LENGTH if length <= (embed-1)*delay, or to
   D2_ERR_RESCALE_ZERO_INTERVAL if rescale is set and some component's raw
   data range is zero (mirroring rescale_data()'s hard-exit case but
   without exiting the process). *error is set to D2_OK on success.

   Caller must free the result with d2_free(). */
D2Result *d2_compute(double *const *series, unsigned long length,
		      unsigned int dim, unsigned int embed,
		      unsigned int delay, unsigned long theiler_window,
		      double eps_max, int eps_max_absolute,
		      double eps_min, int eps_min_absolute,
		      unsigned int howoften, unsigned long maxfound,
		      int rescale, D2Error *error,
		      D2ProgressFn progress, void *user_data);
void d2_free(D2Result *result);

#ifdef __cplusplus
}
#endif

#endif
