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

/* Reentrant API for the av-d2 routine: smooths one dimension-block's worth
   of (eps, y) pairs from a d2 program output file with a centered moving
   average. Extracted out of source_c/av-d2.c so it can be called both from
   the av-d2 CLI and from other bindings (e.g. Python) without going through
   global state or argv parsing. Unlike av-d2.c's main(), this does no file
   I/O or block parsing - it only does the per-block averaging math, taking
   the already-parsed eps[]/y[] pair arrays as input. */

#ifndef _AV_D2_H
#define _AV_D2_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  double *avg_eps;      /* [n_points] window-averaged eps values */
  double *avg_y;         /* [n_points] window-averaged y values */
  unsigned long n_points; /* number of averaged points; 0 if the averaging
			      window (2*aver+1) does not fit within howmany
			      points, in which case avg_eps/avg_y are NULL */
} AvD2Result;

/* Computes the centered moving average of eps[0..howmany-1] and
   y[0..howmany-1] with half-window `aver` (window size 2*aver+1), the same
   way av-d2.c's main() averages one dimension-block: for every k in
   [aver, howmany-aver), avg_eps/avg_y hold the mean of eps/y over
   [k-aver, k+aver].

   Returns NULL if eps or y is NULL, or aver < 0 (invalid input). If aver
   is valid but the window does not fit within howmany points
   (howmany <= 2*aver), returns a result with n_points == 0 rather than the
   out-of-bounds array access av-d2.c's own unsigned-wraparound loop bound
   would suffer from in that case - this is the one place the reentrant
   core intentionally diverges from a literal translation of the CLI loop,
   since a library entry point can be handed arbitrary array lengths that
   the CLI's own d2-file block sizes never exercised.

   Caller must free the result with av_d2_free(). */
AvD2Result *av_d2_average(const double *eps, const double *y,
			   unsigned long howmany, int aver);

void av_d2_free(AvD2Result *result);

#ifdef __cplusplus
}
#endif

#endif
