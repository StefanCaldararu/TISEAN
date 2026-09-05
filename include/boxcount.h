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

/* Reentrant API for the boxcount routine: estimates generalized (Renyi)
   entropies of order Q via a box-partition of the (possibly delay-embedded,
   multivariate) data. Extracted out of source_c/boxcount.c so it can be
   called both from the boxcount CLI and from other bindings (e.g. Python)
   without going through global state, argv parsing, or the process-exiting
   error path in the generic rescale_data() library routine it used to
   call. */

#ifndef _BOXCOUNT_H
#define _BOXCOUNT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  BOXCOUNT_OK = 0,
  BOXCOUNT_ERR_ZERO_INTERVAL  /* some component's raw data range (max-min)
				  is zero */
} BoxCountError;

typedef struct {
  unsigned int dimension;         /* number of components (input rows) */
  unsigned int maxembed;          /* max embedding dimension used */
  unsigned long epscount;         /* number of epsilon scales computed */
  double *eps;                    /* [epscount], epsilon in original
				      (unrescaled) data units - matches the
				      CLI's 1st output column */
  double **entropy;               /* [epscount][dimension*maxembed], the
				      generalized entropy of order q for each
				      (component, embedding) slot at each eps
				      - matches the CLI's 2nd output column.
				      The CLI's 3rd column (successive
				      differences along the flattened slot
				      index) is a display-only derivative of
				      this and is not reproduced here. */
  unsigned int *which_component;  /* [dimension*maxembed], 0-based component
				      index for output slot i */
  unsigned int *which_embed;      /* [dimension*maxembed], 0-based embedding
				      index for output slot i */
} BoxCount;

/* series is [dimension][length]. Computes, for every (component, embedding)
   slot (embedding running 0..maxembed-1, component running 0..dimension-1,
   flattened as embedding*dimension+component - the same order as the CLI's
   which_dims table) and for epscount geometrically-spaced box sizes between
   epsmin and epsmax, the generalized (Renyi) entropy of order q of the
   box-occupation probabilities. delay is the reconstruction delay applied
   between successive embedding coordinates of the same component.

   series is not modified - boxcount_compute rescales a private copy of each
   component to [0,1) (like the CLI's own rescale_data() calls) before
   partitioning it.

   epsmin/epsmax are expressed as fractions of the rescaled [0,1) range
   unless epsmin_absolute/epsmax_absolute is nonzero, in which case they are
   taken to be in the original data units and divided by the largest
   per-component raw range (matching the CLI's -r/-R option semantics,
   where epsminset/epsmaxset gate the same conversion).

   Caller contracts (not checked here - see the Python bindings for
   user-facing validation of them, matching the false_nearest API's
   convention): dimension >= 1, maxembed >= 1, delay >= 1, and
   length > (maxembed-1)*delay (the box partition reads
   series[c][i + embed*delay] for i up to length-(maxembed-1)*delay-1).

   Returns NULL and sets *error (if error is not NULL) to
   BOXCOUNT_ERR_ZERO_INTERVAL if some component's raw data range is zero
   (mirroring rescale_data()'s hard-exit case but without exiting the
   process). *error is set to BOXCOUNT_OK on success.

   Caller must free the result with boxcount_free(). */
BoxCount *boxcount_compute(double *const *series, unsigned long length,
			    unsigned int dimension, unsigned int maxembed,
			    unsigned int delay, double q,
			    double epsmin, int epsmin_absolute,
			    double epsmax, int epsmax_absolute,
			    unsigned int epscount, BoxCountError *error);
void boxcount_free(BoxCount *bc);

#ifdef __cplusplus
}
#endif

#endif
