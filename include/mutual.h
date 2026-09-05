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

/* Reentrant API for the mutual routine: estimates the time-delayed mutual
   information of a data set via an equal-width histogram. Extracted out of
   source_c/mutual.c so it can be called both from the mutual CLI and from
   other bindings (e.g. Python) without going through global state, argv
   parsing, or the process-exiting error path in rescale_data(). */

#ifndef _MUTUAL_H
#define _MUTUAL_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  unsigned long length;  /* length of the input series actually used */
  long partitions;       /* number of histogram bins per dimension (the -b
			     option) */
  long corrlength;       /* maximum lag actually computed (clamped to
			     length-1 if the requested corrlength was too
			     large) */
  double *values;        /* [corrlength+1] conditional-entropy estimate of
			     the mutual information for lags 0..corrlength;
			     values[0] is the plain (lag-0) Shannon entropy */
} MutualResult;

/* Estimates the time-delayed mutual information of series[0..length-1] by
   rescaling it into [0,1) and binning it into `partitions` equal-width
   boxes, then computing the conditional entropy between the series and its
   own lag-t copy for each t in 0..corrlength, the same way the mutual CLI
   does it. series is not modified.

   partitions must be >= 1 and corrlength must be >= 0 - the CLI itself
   never produces values outside those ranges (its -b/-D options are parsed
   as unsigned), so this is documented as a caller contract rather than
   validated here; see the Python bindings for user-facing validation of
   those two.

   Returns NULL if length == 0 or the data is constant (min == max),
   mirroring rescale_data()'s hard-exit case but without exiting the
   process. */
MutualResult *mutual_compute(const double *series, unsigned long length,
			      long partitions, long corrlength);
void mutual_free(MutualResult *result);

#ifdef __cplusplus
}
#endif

#endif
