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

/* Reentrant core of d2, factored out of source_c/d2.c so it has no
   dependency on argv parsing, file-scope globals, wall-clock-triggered
   periodic file dumps, or the process-exiting error paths in the generic
   rescale_data() library routine (and d2.c's own "delay vector too large"
   check) it used to call. The math here (scramble()'s permutation,
   make_c2_dim()/make_c2_1()'s box-assisted neighbour search and the
   epsilon-adaptive-narrowing main loop) is unchanged from the original,
   except: scramble()'s helper arrays are now all freed (the original
   leaked one of them - `scnhelp` - since the CLI process exits right after
   anyway; a reusable library entry point can't afford that), and the
   box-insertion code duplicated at two call sites in main() is shared via
   insert_point() below (still doing the exact same math at each site). */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include "../routines/tsa.h"
#include "../../include/d2.h"

/* Size of the field for box assisted neighbour searching (has to be a
   power of 2) - matches d2.c's own NMAX. */
#define NMAX 256
/* Size of the box for the scramble routine - matches d2.c's own SCBOX. */
#define SCBOX 4096

/* Replaces the global series/scr/box/list/boxc1/listc1/found/epsinv/...
   that make_c2_dim()/make_c2_1() used to close over, so they're reentrant. */
typedef struct {
  double *const *series;   /* [dim][length], possibly rescaled working copy */
  unsigned int dim, embed, delay;
  unsigned long theiler;
  long *scr;                /* [hlength] permutation of point indices */
  long *box;                 /* flat [NMAX*NMAX], box[x*NMAX+y] */
  long *list;                  /* [length] */
  long *boxc1;                  /* [NMAX] */
  long *listc1;                  /* [length] */
  double **found;                /* [n_blocks][howoften] */
  double epsinv, eps_max, eps_min, lneps, lnfac;
  int imax, howoften1, imin;
} D2State;

static void scramble_core(unsigned long length, unsigned int embed,
			   unsigned int delay, long *scr)
{
  long i, j, k, m;
  unsigned long rnd, rndf, hlength, allscr = 0;
  long *scfound, *scnhelp, scnfound;
  long scbox[SCBOX], lswap, element, scbox1 = SCBOX - 1;
  double *rz, *schelp, sceps = (double)SCBOX - 0.001, swap;

  hlength = length - (unsigned long)(embed - 1) * delay;

  if (sizeof(long) == 8) {
    rndf = 13 * 13 * 13 * 13;
    rndf = rndf * rndf * rndf * 13;
    rnd = 0x849178L;
  }
  else {
    rndf = 69069;
    rnd = 0x234571L;
  }
  for (i = 0; i < 1000; i++)
    rnd = rnd * rndf + 1;

  check_alloc(rz = (double *)malloc(sizeof(double) * hlength));
  check_alloc(scfound = (long *)malloc(sizeof(long) * hlength));
  check_alloc(scnhelp = (long *)malloc(sizeof(long) * hlength));
  check_alloc(schelp = (double *)malloc(sizeof(double) * hlength));

  for (i = 0; i < (long)hlength; i++)
    rz[i] = (double)(rnd = rnd * rndf + 1) / ULONG_MAX;

  for (i = 0; i < SCBOX; i++)
    scbox[i] = -1;
  for (i = 0; i < (long)hlength; i++) {
    m = (int)(rz[i] * sceps) & scbox1;
    scfound[i] = scbox[m];
    scbox[m] = i;
  }
  for (i = 0; i < SCBOX; i++) {
    scnfound = 0;
    element = scbox[i];
    while (element != -1) {
      scnhelp[scnfound] = element;
      schelp[scnfound++] = rz[element];
      element = scfound[element];
    }

    for (j = 0; j < scnfound - 1; j++)
      for (k = j + 1; k < scnfound; k++)
	if (schelp[k] < schelp[j]) {
	  swap = schelp[k];
	  schelp[k] = schelp[j];
	  schelp[j] = swap;
	  lswap = scnhelp[k];
	  scnhelp[k] = scnhelp[j];
	  scnhelp[j] = lswap;
	}
    for (j = 0; j < scnfound; j++)
      scr[allscr + j] = scnhelp[j];
    allscr += scnfound;
  }

  free(rz);
  free(scfound);
  free(scnhelp);
  free(schelp);
}

/* x=box[0], y=box[1] coordinates of point sn, matching the two (DIM>1 vs
   DIM==1) branches duplicated at both call sites in d2.c's main(). */
static void point_xy(const D2State *s, long sn, long *x, long *y)
{
  if (s->dim > 1) {
    *x = (long)(s->series[0][sn] * s->epsinv) & s->imax;
    *y = (long)(s->series[1][sn] * s->epsinv) & s->imax;
  }
  else {
    *x = (long)(s->series[0][sn] * s->epsinv) & s->imax;
    *y = (long)(s->series[0][sn + s->delay] * s->epsinv) & s->imax;
  }
}

static void insert_point(D2State *s, long sn)
{
  long x, y;

  point_xy(s, sn, &x, &y);
  s->list[sn] = s->box[x * NMAX + y];
  s->box[x * NMAX + y] = sn;
  s->listc1[sn] = s->boxc1[x];
  s->boxc1[x] = sn;
}

static void make_c2_dim_core(D2State *s, long n)
{
  char small;
  long i, j, k, x, y, i1, i2, j1, element, n1, maxi, count, hi;
  double *hs, max, dx;

  check_alloc(hs = (double *)malloc(sizeof(double) * s->embed * s->dim));
  n1 = s->scr[n];

  count = 0;
  for (i1 = 0; i1 < (long)s->embed; i1++) {
    i2 = i1 * s->delay;
    for (j = 0; j < (long)s->dim; j++)
      hs[count++] = s->series[j][n1 + i2];
  }

  x = (long)(hs[0] * s->epsinv) & s->imax;
  y = (long)(hs[1] * s->epsinv) & s->imax;

  for (i1 = x - 1; i1 <= x + 1; i1++) {
    i2 = i1 & s->imax;
    for (j1 = y - 1; j1 <= y + 1; j1++) {
      element = s->box[i2 * NMAX + (j1 & s->imax)];
      while (element != -1) {
	if (labs((long)(element - n1)) > (long)s->theiler) {
	  count = 0;
	  max = 0.0;
	  maxi = s->howoften1;
	  small = 0;
	  for (i = 0; i < (long)s->embed; i++) {
	    hi = i * s->delay;
	    for (j = 0; j < (long)s->dim; j++) {
	      dx = fabs(hs[count] - s->series[j][element + hi]);
	      if (dx <= s->eps_max) {
		if (dx > max) {
		  max = dx;
		  if (max < s->eps_min)
		    maxi = s->howoften1;
		  else
		    maxi = (long)((s->lneps - log(max)) / s->lnfac);
		}
		if (count > 0)
		  for (k = s->imin; k <= maxi; k++)
		    s->found[count][k] += 1.0;
	      }
	      else {
		small = 1;
		break;
	      }
	      count++;
	    }
	    if (small)
	      break;
	  }
	}
	element = s->list[element];
      }
    }
  }

  free(hs);
}

static void make_c2_1_core(D2State *s, long n)
{
  int i, x, i1, maxi;
  long element, n1;
  double hs, max;

  n1 = s->scr[n];
  hs = s->series[0][n1];

  x = (int)(hs * s->epsinv) & s->imax;

  for (i1 = x - 1; i1 <= x + 1; i1++) {
    element = s->boxc1[i1 & s->imax];
    while (element != -1) {
      if (labs(element - n1) > (long)s->theiler) {
	max = fabs(hs - s->series[0][element]);
	if (max <= s->eps_max) {
	  if (max < s->eps_min)
	    maxi = s->howoften1;
	  else
	    maxi = (int)((s->lneps - log(max)) / s->lnfac);
	  for (i = s->imin; i <= maxi; i++)
	    s->found[0][i] += 1.0;
	}
      }
      element = s->listc1[element];
    }
  }
}

/* Allocates a D2Result shell (all leaf arrays present, contents
   uninitialized) for the given shape - shared by the live progress
   snapshot (built once per center point when a progress callback is given)
   and the final result (built once, after the loop, when it is not). */
static D2Result *alloc_result_shell(unsigned int dim, unsigned int embed,
				     unsigned int n_blocks,
				     unsigned int howoften)
{
  unsigned int blk;
  D2Result *result;

  check_alloc(result = (D2Result *)malloc(sizeof(D2Result)));
  result->dim = dim;
  result->embed = embed;
  result->n_blocks = n_blocks;
  result->howoften = howoften;
  check_alloc(result->eps = (double *)malloc(sizeof(double) * howoften));
  check_alloc(result->c2 = (double **)malloc(sizeof(double *) * n_blocks));
  check_alloc(result->h2 = (double **)malloc(sizeof(double *) * n_blocks));
  check_alloc(result->d2 = (double **)malloc(sizeof(double *) * n_blocks));
  for (blk = 0; blk < n_blocks; blk++) {
    check_alloc(result->c2[blk] = (double *)malloc(sizeof(double) * howoften));
    check_alloc(result->h2[blk] = (double *)malloc(sizeof(double) * howoften));
    check_alloc(result->d2[blk] = (double *)malloc(sizeof(double) * howoften));
  }
  return result;
}

/* Fills an already-allocated (see alloc_result_shell()) D2Result from the
   current found[][]/norm[] state - the same formulas as d2.c's own
   periodic/final dump block (see d2.h for the NaN-gating this replaces the
   original's conditional fprintf()s with). */
static void fill_result_tables(D2Result *out, const D2State *s,
				const double *epsm, const double *norm,
				double lnfac)
{
  unsigned int blk;
  unsigned long j;

  for (j = 0; j < out->howoften; j++)
    out->eps[j] = epsm[j];

  for (blk = 0; blk < out->n_blocks; blk++) {
    for (j = 0; j < out->howoften; j++) {
      out->c2[blk][j] = (norm[j] > 0.0) ? s->found[blk][j] / norm[j] : NAN;

      if (blk == 0)
	out->h2[blk][j] =
	  (s->found[0][j] > 0.0) ? -log(s->found[0][j] / norm[j]) : NAN;
      else
	out->h2[blk][j] =
	  (s->found[blk - 1][j] > 0.0 && s->found[blk][j] > 0.0) ?
	  log(s->found[blk - 1][j] / s->found[blk][j]) : NAN;

      out->d2[blk][j] = NAN;
    }
    for (j = 1; j < out->howoften; j++) {
      if (s->found[blk][j] > 0.0 && s->found[blk][j - 1] > 0.0)
	out->d2[blk][j] = log(s->found[blk][j - 1] / s->found[blk][j]
			       / norm[j - 1] * norm[j]) / lnfac;
    }
  }
}

D2Result *d2_compute(double *const *series_in, unsigned long length,
		      unsigned int dim, unsigned int embed,
		      unsigned int delay, unsigned long theiler_window,
		      double eps_max, int eps_max_absolute,
		      double eps_min, int eps_min_absolute,
		      unsigned int howoften, unsigned long maxfound,
		      int rescale, D2Error *error,
		      D2ProgressFn progress, void *user_data)
{
  unsigned int i, n_blocks;
  unsigned long j, hlength, nmax;
  long i1, j1, n, sn, n1, n2, lnorm, maxembed_idx;
  int smaller;
  double **work;
  double min, interval, maxinterval;
  double epsfactor, lneps, lnfac;
  double *epsm, *norm;
  long *oscr;
  D2State s;
  D2Result *result;

  if (error != NULL)
    *error = D2_OK;

  if (maxfound == 0)
    maxfound = ULONG_MAX;

  if ((long)(length - (unsigned long)(embed - 1) * delay) <= 0) {
    if (error != NULL)
      *error = D2_ERR_VECTOR_TOO_LARGE_FOR_LENGTH;
    return NULL;
  }

  /* Private working copy - d2_compute must not mutate the caller's data
     the way the CLI's in-place rescale_data() call does. */
  check_alloc(work = (double **)malloc(sizeof(double *) * dim));
  for (i = 0; i < dim; i++) {
    check_alloc(work[i] = (double *)malloc(sizeof(double) * length));
    for (j = 0; j < length; j++)
      work[i][j] = series_in[i][j];
  }

  if (rescale) {
    for (i = 0; i < dim; i++) {
      /* rescale_data(work[i], length, &min, &interval), inlined so a
	 zero-range component returns an error instead of exiting. */
      min = interval = work[i][0];
      for (j = 1; j < length; j++) {
	if (work[i][j] < min) min = work[i][j];
	if (work[i][j] > interval) interval = work[i][j];
      }
      interval -= min;
      if (interval == 0.0) {
	for (j = 0; j < dim; j++)
	  free(work[j]);
	free(work);
	if (error != NULL)
	  *error = D2_ERR_RESCALE_ZERO_INTERVAL;
	return NULL;
      }
      for (j = 0; j < length; j++)
	work[i][j] = (work[i][j] - min) / interval;
    }
    maxinterval = 1.0;
  }
  else {
    maxinterval = 0.0;
    for (i = 0; i < dim; i++) {
      min = interval = work[i][0];
      for (j = 1; j < length; j++) {
	if (min > work[i][j]) min = work[i][j];
	if (interval < work[i][j]) interval = work[i][j];
      }
      interval -= min;
      if (interval > maxinterval)
	maxinterval = interval;
    }
  }

  if (!eps_max_absolute)
    eps_max *= maxinterval;
  if (!eps_min_absolute)
    eps_min *= maxinterval;
  eps_max = (fabs(eps_max) < maxinterval) ? fabs(eps_max) : maxinterval;
  eps_min = (fabs(eps_min) < eps_max) ? fabs(eps_min) : eps_max / 2.;

  n_blocks = dim * embed;
  maxembed_idx = (long)n_blocks - 1;
  hlength = length - (unsigned long)(embed - 1) * delay;
  nmax = hlength;

  check_alloc(s.list = (long *)malloc(length * sizeof(long)));
  check_alloc(s.listc1 = (long *)malloc(length * sizeof(long)));
  check_alloc(s.scr = (long *)malloc(sizeof(long) * hlength));
  check_alloc(oscr = (long *)malloc(sizeof(long) * hlength));
  check_alloc(s.box = (long *)malloc(sizeof(long) * NMAX * NMAX));
  check_alloc(s.boxc1 = (long *)malloc(sizeof(long) * NMAX));
  check_alloc(s.found = (double **)malloc(n_blocks * sizeof(double *)));
  for (i = 0; i < n_blocks; i++)
    check_alloc(s.found[i] = (double *)malloc(howoften * sizeof(double)));
  check_alloc(norm = (double *)malloc(howoften * sizeof(double)));
  check_alloc(epsm = (double *)malloc(howoften * sizeof(double)));

  s.dim = dim;
  s.embed = embed;
  s.delay = delay;
  s.theiler = theiler_window;
  s.series = work;
  s.imax = NMAX - 1;
  s.howoften1 = (int)howoften - 1;
  s.eps_max = eps_max;
  s.eps_min = eps_min;
  s.epsinv = 1.0 / eps_max;
  epsfactor = pow(eps_max / eps_min, 1.0 / (double)s.howoften1);
  lneps = log(eps_max);
  lnfac = log(epsfactor);
  s.lneps = lneps;
  s.lnfac = lnfac;
  s.imin = 0;

  epsm[0] = eps_max;
  norm[0] = 0.0;
  for (i = 1; i < howoften; i++) {
    norm[i] = 0.0;
    epsm[i] = epsm[i - 1] / epsfactor;
  }

  scramble_core(length, embed, delay, s.scr);
  for (i1 = 0; (unsigned long)i1 < hlength; i1++)
    oscr[s.scr[i1]] = i1;

  for (i = 0; i < n_blocks; i++)
    for (j = 0; j < howoften; j++)
      s.found[i][j] = 0.0;

  for (i1 = 0; i1 < NMAX; i1++) {
    s.boxc1[i1] = -1;
    for (j1 = 0; j1 < NMAX; j1++)
      s.box[i1 * NMAX + j1] = -1;
  }

  if (progress != NULL)
    result = alloc_result_shell(dim, embed, n_blocks, howoften);
  else
    result = NULL;

  for (n = 1; (unsigned long)n < nmax; n++) {
    smaller = 0;
    sn = s.scr[n - 1];
    insert_point(&s, sn);

    i1 = s.imin;
    while (s.found[maxembed_idx][i1] >= (double)maxfound) {
      smaller = 1;
      if (++i1 > s.howoften1)
	break;
    }
    if (smaller) {
      s.imin = (int)i1;
      if (s.imin <= s.howoften1) {
	s.eps_max = epsm[s.imin];
	s.epsinv = 1.0 / s.eps_max;
	for (i1 = 0; i1 < NMAX; i1++) {
	  s.boxc1[i1] = -1;
	  for (j1 = 0; j1 < NMAX; j1++)
	    s.box[i1 * NMAX + j1] = -1;
	}
	for (i1 = 0; i1 < n; i1++)
	  insert_point(&s, s.scr[i1]);
      }
    }

    if (s.imin <= s.howoften1) {
      lnorm = n;
      if (theiler_window > 0) {
	sn = s.scr[n];
	n1 = (sn - (long)theiler_window >= 0) ? sn - (long)theiler_window : 0;
	n2 = (sn + (long)theiler_window < (long)hlength) ?
	  sn + (long)theiler_window : (long)hlength - 1;
	for (i1 = n1; i1 <= n2; i1++)
	  if (oscr[i1] < n)
	    lnorm--;
      }

      if (embed * dim > 1)
	make_c2_dim_core(&s, n);
      make_c2_1_core(&s, n);
      for (i1 = s.imin; i1 < (long)howoften; i1++)
	norm[i1] += (double)lnorm;
    }

    /* Mirrors the CLI's own dump condition
       "(time-lasttime>WHEN) || (n==(nmax-1)) || (imin>howoften1)": called
       unconditionally every center point (like lyap_spec's own progress
       callback), leaving the wall-clock half of that OR to the caller -
       is_last covers the other two (both cause the loop below to stop
       right after this call, exactly like the CLI's dump-then-exit(0)). */
    if (progress != NULL) {
      int is_last = ((unsigned long)n == nmax - 1) || (s.imin > s.howoften1);
      double current_eps_max = (s.imin <= s.howoften1) ?
	epsm[s.imin] : epsm[s.howoften1];

      fill_result_tables(result, &s, epsm, norm, lnfac);
      progress(result, (unsigned long)n, current_eps_max, is_last, user_data);
    }

    if (s.imin > s.howoften1)
      break;
  }

  /* Build the final output tables from the final found[][]/norm[] state if
     they were not already built (and hasn't already returned via the
     progress callback above) - mirroring the CLI's own last dump, which
     always reflects the state as of the final n reached, whether via
     natural loop completion (n == nmax-1) or the imin > howoften1 early
     stop (which the CLI reaches via the same dump-then-exit(0) path the
     loop above mirrors by breaking instead of exiting). */
  if (result == NULL) {
    result = alloc_result_shell(dim, embed, n_blocks, howoften);
    fill_result_tables(result, &s, epsm, norm, lnfac);
  }

  free(s.list);
  free(s.listc1);
  free(s.scr);
  free(oscr);
  free(s.box);
  free(s.boxc1);
  for (i = 0; i < n_blocks; i++)
    free(s.found[i]);
  free(s.found);
  free(norm);
  free(epsm);
  for (i = 0; i < dim; i++)
    free(work[i]);
  free(work);

  return result;
}

void d2_free(D2Result *result)
{
  unsigned int i;

  if (result == NULL)
    return;
  free(result->eps);
  if (result->c2 != NULL) {
    for (i = 0; i < result->n_blocks; i++)
      free(result->c2[i]);
    free(result->c2);
  }
  if (result->h2 != NULL) {
    for (i = 0; i < result->n_blocks; i++)
      free(result->h2[i]);
    free(result->h2);
  }
  if (result->d2 != NULL) {
    for (i = 0; i < result->n_blocks; i++)
      free(result->d2[i]);
    free(result->d2);
  }
  free(result);
}
