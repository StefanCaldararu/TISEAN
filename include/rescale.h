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

/* Reentrant API for the rescale routine: rescales each row of a data set,
   either to a given [xmin,xmax) interval, to zero mean, to unit variance, or
   both of the latter two at once. Extracted out of source_c/rescale.c so it
   can be called both from the rescale CLI and from other bindings (e.g.
   Python) without going through global state, argv parsing, or the
   process-exiting error path in the generic variance()/rescale_data()
   library routines it used to call. */

#ifndef _RESCALE_H
#define _RESCALE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  RESCALE_OK = 0,
  RESCALE_ERR_EMPTY_SERIES,
  RESCALE_ERR_WRONG_INTERVAL,
  RESCALE_ERR_ZERO_VARIANCE
} RescaleError;

typedef struct {
  unsigned int dim;      /* number of rows/components */
  unsigned long length;  /* number of points per row */
  double **data;          /* [dim][length] rescaled series */
} RescaleResult;

/* Rescales each of the dim rows of series[0..dim-1][0..length-1]
   independently, matching the rescale CLI's -a/-v/-z/-Z options:
     - if set_av is non-zero, the row's own mean is subtracted
     - if set_var is non-zero, the row is divided by its own standard
       deviation
     - if both set_av and set_var are zero, the row is linearly rescaled
       from its own [min,max] onto [xmin,xmax) instead (the CLI's default
       mode); xmin/xmax are otherwise ignored
   Passing both set_av and set_var non-zero applies both (subtract the mean,
   then divide by the standard deviation), exactly like the CLI's -a -v
   combination. series is not modified.

   Returns NULL and sets *error (if error is non-NULL) if:
     - length == 0 (RESCALE_ERR_EMPTY_SERIES)
     - xmin >= xmax (RESCALE_ERR_WRONG_INTERVAL) - checked unconditionally,
       even when set_av or set_var is non-zero and xmin/xmax would otherwise
       go unused, matching a quirk of the CLI where this check runs before
       the per-row mode is applied
     - some row of series is constant, i.e. has zero variance
       (RESCALE_ERR_ZERO_VARIANCE) - the CLI's variance() check runs
       unconditionally on every row too, before the mode-specific
       rescaling, so even the plain min/max mode (set_av == set_var == 0)
       rejects a constant row via this check rather than ever reaching its
       own interior zero-interval check in rescale_data()

   Caller must free a non-NULL result with rescale_free(). */
RescaleResult *rescale_compute(double *const *series, unsigned long length,
				unsigned int dim, int set_av, int set_var,
				double xmin, double xmax, RescaleError *error);
void rescale_free(RescaleResult *result);

#ifdef __cplusplus
}
#endif

#endif
