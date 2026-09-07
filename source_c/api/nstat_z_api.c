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

/* Reentrant core of nstat_z, factored out of source_c/nstat_z.c so it has
   no dependency on argv parsing, file-scope globals, or the process-exiting
   error paths in the generic variance()/rescale_data() library routines it
   used to call. The math here (the global rescale, the per-piece variance,
   the window-matrix construction from first/second selections and offsets,
   the box-assisted neighbor search and the zeroth-order fit in
   make_fit()/main()) is unchanged from the original. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../routines/tsa.h"
#include "../../include/nstat_z.h"

/*number of boxes for the neighbor search algorithm*/
#define NMAX 128

static double make_fit(const double *series1, const double *series2,
			const unsigned long *found, unsigned long number,
			unsigned long step, long act)
{
  double casted = 0.0;
  const double *help = series1 + step;
  unsigned long i;

  for (i = 0; i < number; i++)
    casted += help[found[i]];
  casted /= (double)number;

  return sqr(casted - series2[act + step]);
}

NstatZ *nstat_z_compute(const double *series_in, unsigned long length,
			 unsigned int pieces,
			 const char *first_window, const char *second_window,
			 int first_offset, int second_offset,
			 unsigned int dim, unsigned int delay,
			 unsigned int minn, unsigned long step,
			 unsigned long causal, unsigned long center,
			 char centerset, double eps0, char epsset,
			 double epsf, NstatZError *error)
{
  unsigned long i, clength, pstart, n_pairs, idx;
  long first, second;
  double *series, min, interval;
  double *rms;
  char **window;
  long *list;
  unsigned long *found, *hfound, actfound;
  char *done, alldone;
  long **box;
  double epsilon, err_sum, *series1, *series2;
  unsigned int *result_first, *result_second;
  double *result_value;
  NstatZ *result;

  if (error != NULL)
    *error = NSTAT_Z_OK;

  /* rescale_data(series, length, &min, &interval), on a private copy */
  check_alloc(series = (double *)malloc(sizeof(double) * length));
  memcpy(series, series_in, sizeof(double) * length);
  min = interval = series[0];
  for (i = 1; i < length; i++) {
    if (series[i] < min) min = series[i];
    if (series[i] > interval) interval = series[i];
  }
  interval -= min;
  if (interval == 0.0) {
    free(series);
    if (error != NULL)
      *error = NSTAT_Z_ERR_ZERO_INTERVAL;
    return NULL;
  }
  for (i = 0; i < length; i++)
    series[i] = (series[i] - min) / interval;

  if (epsset)
    eps0 /= interval;

  /* Same arithmetic as the original (clength, dim, delay, step, minn were
     unsigned long/int mixed in unsigned arithmetic there too), including
     its unsigned-wraparound behavior for pathologically large `pieces`. */
  clength = (length - (unsigned long)(dim - 1) * (unsigned long)delay) / pieces;
  if ((clength - (unsigned long)(dim - 1) * (unsigned long)delay - step) <
      (unsigned long)minn) {
    free(series);
    if (error != NULL)
      *error = NSTAT_Z_ERR_TOO_MANY_PIECES;
    return NULL;
  }

  /* variance(series+i*clength, clength, &av, &rms[i]), per piece, on the
     already (globally) rescaled data */
  check_alloc(rms = (double *)malloc(sizeof(double) * pieces));
  for (i = 0; i < pieces; i++) {
    double *piece = series + i * clength;
    double av = 0.0, var = 0.0, h;
    unsigned long k;

    for (k = 0; k < clength; k++) {
      h = piece[k];
      av += h;
      var += h * h;
    }
    av /= (double)clength;
    var = sqrt(fabs(var / (double)clength - av * av));
    if (var == 0.0) {
      free(series);
      free(rms);
      if (error != NULL)
	*error = NSTAT_Z_ERR_ZERO_VARIANCE;
      return NULL;
    }
    rms[i] = var;
  }

  pstart = (unsigned long)(dim - 1) * delay;
  if (!centerset)
    center = clength - step;
  else
    center = (center < (clength - step - pstart)) ? center : clength - step - pstart;

  /* Build the pieces x pieces window matrix from first_window/second_window
     and first_offset/second_offset, matching the CLI main()'s pre-refactor
     logic exactly (this part never called exit()). */
  check_alloc(window = (char **)malloc(sizeof(char *) * pieces));
  for (first = 0; first < (long)pieces; first++)
    check_alloc(window[first] = (char *)malloc(pieces));

  /* A "+N" offset for one side (parse_offset() in the CLI) implies that
     side's plain window selection is all-zero: the CLI's scan_options()
     only ever calls parse_offset() (which zeroes the array) or parse_out()
     (which fills it from a range spec) for a given -1/-2, never both. */
  for (first = 0; first < (long)pieces; first++)
    for (second = 0; second < (long)pieces; second++)
      window[first][second] =
	  (first_offset != -1 ? 0 : (first_window ? first_window[first] : 1)) &&
	  (second_offset != -1 ? 0 : (second_window ? second_window[second] : 1));
  if (first_offset != -1) {
    for (second = 0; second < (long)pieces; second++)
      for (first = second - first_offset; first <= second + first_offset; first++)
	if ((first >= 0) && (first < (long)pieces))
	  window[first][second] = second_window ? second_window[second] : 1;
  }
  if (second_offset != -1) {
    for (first = 0; first < (long)pieces; first++)
      for (second = first - second_offset; second <= first + second_offset; second++)
	if ((second >= 0) && (second < (long)pieces))
	  window[first][second] = first_window ? first_window[first] : 1;
  }

  n_pairs = 0;
  for (first = 0; first < (long)pieces; first++)
    for (second = 0; second < (long)pieces; second++)
      if (window[first][second])
	n_pairs++;

  result_first = NULL;
  result_second = NULL;
  result_value = NULL;
  if (n_pairs > 0) {
    check_alloc(result_first = (unsigned int *)malloc(sizeof(unsigned int) * n_pairs));
    check_alloc(result_second = (unsigned int *)malloc(sizeof(unsigned int) * n_pairs));
    check_alloc(result_value = (double *)malloc(sizeof(double) * n_pairs));
  }

  check_alloc(list = (long *)malloc(sizeof(long) * length));
  check_alloc(found = (unsigned long *)malloc(sizeof(long) * length));
  check_alloc(hfound = (unsigned long *)malloc(sizeof(long) * length));
  check_alloc(done = (char *)malloc(sizeof(char) * length));
  check_alloc(box = (long **)malloc(sizeof(long *) * NMAX));
  for (i = 0; i < NMAX; i++)
    check_alloc(box[i] = (long *)malloc(sizeof(long) * NMAX));

  idx = 0;
  for (first = 0; first < (long)pieces; first++) {
    for (second = 0; second < (long)pieces; second++) {
      if (!window[first][second])
	continue;

      series1 = series + first * clength;
      series2 = series + second * clength;
      for (i = 0; i < length; i++)
	done[i] = 0;
      alldone = 0;
      epsilon = eps0 / epsf;
      err_sum = 0.0;
      while (!alldone) {
	alldone = 1;
	epsilon *= epsf;
	make_box(series1, box, list, clength - step, NMAX, dim, delay, epsilon);
	for (i = pstart; i < pstart + center; i++)
	  if (!done[i]) {
	    actfound = find_neighbors(series1, box, list, series2 + i, clength,
				       NMAX, dim, delay, epsilon, hfound);
	    actfound = exclude_interval(actfound, (long)i - (long)causal + 1,
					 (long)i + (long)causal + (long)pstart - 1,
					 hfound, found);
	    if (actfound >= minn) {
	      err_sum += make_fit(series1, series2, found, actfound, step, (long)i);
	      done[i] = 1;
	    }
	    alldone &= done[i];
	  }
      }

      result_first[idx] = (unsigned int)first;
      result_second[idx] = (unsigned int)second;
      result_value[idx] = sqrt(err_sum / (double)center) / rms[second];
      idx++;
    }
  }

  free(list);
  free(found);
  free(hfound);
  free(done);
  for (i = 0; i < NMAX; i++)
    free(box[i]);
  free(box);
  for (first = 0; first < (long)pieces; first++)
    free(window[first]);
  free(window);
  free(rms);
  free(series);

  check_alloc(result = (NstatZ *)malloc(sizeof(NstatZ)));
  result->pieces = pieces;
  result->n_pairs = n_pairs;
  result->first = result_first;
  result->second = result_second;
  result->value = result_value;

  return result;
}

void nstat_z_free(NstatZ *result)
{
  if (result == NULL)
    return;
  free(result->first);
  free(result->second);
  free(result->value);
  free(result);
}
