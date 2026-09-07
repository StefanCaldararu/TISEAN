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

/* Reentrant API for the nrlazy routine: simple multivariate nonlinear noise
   reduction (each embedded point is replaced by the average of its
   neighbors in delay-embedding space). Extracted out of source_c/nrlazy.c
   so it can be called both from the nrlazy CLI and from other bindings
   (e.g. Python) without going through global state, argv parsing, or the
   process-exiting error path in rescale_data(). */

#ifndef _NRLAZY_H
#define _NRLAZY_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  unsigned int comp;
  unsigned long length;
  double **series;         /* [comp][length]; corrected data, scaled back to
			       the original (raw) units of the input series */
  unsigned int *neighbors; /* [length]; number of neighbors found for each
			       point during the LAST iteration (see the
			       `iterations` argument below). Entries before
			       (embed-1)*delay, which the neighbor search
			       never visits, are always 1. */
} NRLazyResult;

/* Optional callback invoked once after each of nrlazy_correct()'s
   `iterations` passes, with that pass's series/neighbors in the same shape
   and units as NRLazyResult's fields (i.e. already scaled back to the
   original units) - lets a caller (e.g. the CLI, which writes every
   intermediate iteration to its own file when -V's VER_USR1 bit is set)
   observe every pass, not just the last one, without the reentrant core
   having to expose any rescaling/eps internals. Ignored if NULL. */
typedef void (*NRLazyIterationFn)(unsigned int iter,unsigned int iterations,
				    double *const *series,
				    const unsigned int *neighbors,
				    void *user_data);

/* Performs simple multivariate nonlinear noise reduction, mirroring
   source_c/nrlazy.c's main()/correct(). series is [comp][length] and holds
   the RAW (unscaled) input; it is not modified. Internally, nrlazy_correct()
   rescales a private copy of each component to [0,1] the same way
   rescale_data() does it (min-max normalization), corrects that rescaled
   copy for `iterations` passes (each pass replaces every embedded point by
   the average of the neighbors found within `eps` of it, using
   make_multi_index()/make_multi_box2() for the delay embedding and neighbor
   search), then writes the result back scaled to the original units.

   eps_r is the -r flag's raw value (a fraction of the largest per-component
   data interval), or NaN if -r was not given -- in that case eps defaults
   to a plain 1.e-3 in the already-rescaled [0,1) space (NOT
   (data interval)/1000, despite what the CLI's help text says: that
   mismatch is a quirk of the original code, preserved here as-is). eps_v is
   the -v flag's value (the neighborhood radius in units of the largest
   per-component standard deviation of the rescaled data); if not NaN, it
   overwrites whatever eps_r would have produced, matching the CLI's "-v
   overwrites -r".

   Returns NULL if any component of series is constant (interval == 0),
   mirroring rescale_data()'s hard-exit case but without exiting the
   process; if bad_value is non-NULL, *bad_value is set to that component's
   constant value (matching what rescale_data()'s "data ranges from %e to
   %e" message would have printed - both %e are that same value).
   variance()'s own hard-exit case (zero standard deviation) can never
   actually trigger here: it is only evaluated after rescale_data()
   succeeds, at which point the rescaled data can't be constant, so its
   standard deviation can't be zero either.

   comp, embed, delay and iterations are not validated here, mirroring the
   original CLI (which never validates them either): passing comp == 0,
   embed == 0, delay == 0, iterations == 0, or (embed-1)*delay >= length
   reproduces whatever the original code would do for those inputs
   (typically a silent no-op correction; iterations == 0 additionally
   leaves `neighbors` holding unspecified values, since the original never
   initializes its equivalent array in that case either). Callers that want
   a clean error for these should validate before calling.

   on_iteration/on_iteration_data: see NRLazyIterationFn above; pass
   NULL/NULL to ignore intermediate iterations. */
NRLazyResult *nrlazy_correct(double *const *series, unsigned long length,
			      unsigned int comp, unsigned int embed,
			      unsigned int delay, unsigned int iterations,
			      double eps_r, double eps_v, double *bad_value,
			      NRLazyIterationFn on_iteration,
			      void *on_iteration_data);
void nrlazy_free(NRLazyResult *result);

#ifdef __cplusplus
}
#endif

#endif
