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

/* Reentrant API for the poincare routine: finds Poincare-section crossings
   of a time-delay embedding of a series. Extracted out of
   source_c/poincare.c so it can be called both from the poincare CLI and
   from other bindings (e.g. Python) without going through global state,
   argv parsing, or the process-exiting error path in the generic
   variance() library routine it used to call. */

#ifndef _POINCARE_H
#define _POINCARE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  POINCARE_OK = 0,
  POINCARE_ERR_EMPTY_SERIES,
  POINCARE_ERR_ZERO_VARIANCE,
  POINCARE_ERR_WRONG_COMPONENT,
  POINCARE_ERR_OUTSIDE_REGION
} PoincareError;

typedef struct {
  unsigned long count;  /* number of section crossings recorded (the very
			    first crossing found only seeds the time
			    reference and is not counted/output, matching
			    the CLI) */
  unsigned int dim;      /* embedding dimension; every crossing has dim-1
			    coordinates (component `comp` itself is pinned
			    to `where` and excluded). NULL point/dt below if
			    count == 0; point is also NULL if dim <= 1 (no
			    coordinates to store), even if count > 0 */
  double *point;          /* [count*(dim-1)] flattened, row-major:
			    point[i*(dim-1)+k] is the k-th coordinate (in
			    ascending order of the embedded component
			    index, skipping `comp` itself) of the i-th
			    crossing */
  double *dt;              /* [count] time since the previous crossing */
} PoincareResult;

/* Finds Poincare-section crossings of the (dim,delay)-embedding of
   series[0..length-1], cut through component comp (1-based, must be <=
   dim, matching the CLI's -q) at level `where`: a crossing from below is
   detected wherever series[i] < where <= series[i+1] (dir == 0), or from
   above wherever series[i] > where >= series[i+1] (dir == 1), matching the
   CLI's -C. The very first crossing found only seeds the time reference
   and produces no output row, exactly like the CLI.

   If whereset == 0, `where` is ignored and the series' own mean (a plain
   sequential-sum mean, matching variance()) is used instead, mirroring the
   CLI's -a default.

   If out_min/out_max are non-NULL, they are set to series' own [min,max]
   whenever that has actually been computed - i.e. on every call that
   doesn't fail with POINCARE_ERR_EMPTY_SERIES or POINCARE_ERR_ZERO_VARIANCE,
   even if the call goes on to fail for a different reason. This lets a
   caller reproduce the CLI's own "You want to cut outside the data
   interval" error message without recomputing min/max itself.

   Returns NULL and sets *error (if error is non-NULL) if:
     - length == 0 (POINCARE_ERR_EMPTY_SERIES)
     - series has zero variance (POINCARE_ERR_ZERO_VARIANCE)
     - comp > dim (POINCARE_ERR_WRONG_COMPONENT)
     - the resolved where value falls outside [min(series), max(series)]
       (POINCARE_ERR_OUTSIDE_REGION)

   Caller must free a non-NULL result with poincare_free(). */
PoincareResult *poincare_compute(const double *series, unsigned long length,
				  int dim, int comp, int delay, int dir,
				  int whereset, double where,
				  double *out_min, double *out_max,
				  PoincareError *error);
void poincare_free(PoincareResult *result);

#ifdef __cplusplus
}
#endif

#endif
