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

/* Reentrant API for the recurr routine: finds recurrence-plot neighbor
   pairs in (possibly multivariate, delay-embedded) data. Extracted out of
   source_c/recurr.c so it can be called both from the recurr CLI and from
   other bindings (e.g. Python) without going through global state, argv
   parsing, or the process-exiting error path in rescale_data(). */

#ifndef _RECURR_H
#define _RECURR_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  unsigned long count;  /* number of neighbor pairs found */
  long *point;           /* [count] 1-based index of the scanned point */
  long *neighbor;         /* [count] 1-based index of its recurrence neighbor */
} RecurrResult;

/* Finds recurrence-plot neighbor pairs in series (shape [dim][length]),
   mirroring source_c/recurr.c's main()/lfind_neighbors(). Each dimension
   of series is independently rescaled to [0,1) the same way rescale_data()
   does it, but without exiting the process on a constant dimension: if any
   dimension is constant (max == min), NULL is returned and, if bad_value
   is non-NULL, *bad_value is set to that dimension's constant value
   (matching what rescale_data()'s "data ranges from %e to %e" message
   would have printed - both %e are that same value). series is not
   modified.

   If eps_is_raw is non-zero, eps is interpreted in the original (raw) data
   units and divided by the largest of the per-dimension raw intervals
   before use, matching the CLI's -r flag; if zero, eps is used as-is in
   the already-rescaled [0,1) space, matching the CLI's default (a plain
   1.e-3, i.e. "(data interval)/1000").

   For every scanned point n in (embed-1)*delay..length-1, every other
   point `element` (< n) within Chebyshev distance eps across all `embed`
   delayed copies of all `dim` rescaled components is a candidate neighbor;
   each candidate is kept with probability `fraction`, drawn from the same
   fixed-seed rnd69069() generator the CLI uses (rnd_init() only actually
   reseeds once per process - see routines/rand.c - so results depend on
   any earlier RNG use within the same process, exactly like the CLI).

   Returns NULL (without touching *bad_value) if dim == 0 or length == 0. */
RecurrResult *recurr_find(double *const *series, unsigned long length,
			   unsigned int dim, unsigned int embed,
			   unsigned int delay, double eps, char eps_is_raw,
			   double fraction, double *bad_value);
void recurr_free(RecurrResult *result);

#ifdef __cplusplus
}
#endif

#endif
