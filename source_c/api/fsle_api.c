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

/* Reentrant core of fsle, factored out of source_c/fsle.c so it has no
   dependency on argv parsing, file-scope globals, or the process-exiting
   error paths in the generic variance()/rescale_data() library routines it
   used to call. The math here (the box-assisted nearest-neighbor search in
   put_in_boxes()/make_iterate() and the exponential growth of the epsilon
   bins in main()) is unchanged from the original. */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../routines/tsa.h"
#include "../../include/fsle.h"

#define FSLE_NMAX 256

typedef struct {
  double time, factor, eps;
  long count;
} FSLEBin;

typedef struct {
  const double *series;
  long **box;
  long *list;
  unsigned int dim, delay, mindist;
  unsigned int nmax;
  unsigned long length;
  double eps, epsinv, eps0, epsfactor;
  int howmany;
  FSLEBin *data;
} FSLEState;

static void put_in_boxes(FSLEState *st)
{
  int i,j,x,y,del;
  unsigned long length=st->length;
  unsigned int delay=st->delay,dim=st->dim,nmax=st->nmax;
  const double *series=st->series;
  long **box=st->box,*list=st->list;
  double epsinv=st->epsinv;

  for (i=0;i<FSLE_NMAX;i++)
    for (j=0;j<FSLE_NMAX;j++)
      box[i][j]= -1;

  del=delay*(dim-1);
  for (i=0;i<length-del;i++) {
    x=(int)(series[i]*epsinv)&nmax;
    y=(int)(series[i+del]*epsinv)&nmax;
    list[i]=box[x][y];
    box[x][y]=i;
  }
}

static char make_iterate(FSLEState *st, long act)
{
  char ok=0;
  int x,y,i,j,i1,k,del1=st->dim*st->delay,which;
  long element,minelement= -1;
  double dx=0.0,mindx=2.0,stime;
  const double *series=st->series;
  long **box=st->box,*list=st->list;
  unsigned int nmax=st->nmax,mindist=st->mindist,delay=st->delay;
  unsigned long length=st->length;
  double eps=st->eps,epsinv=st->epsinv,eps0=st->eps0,epsfactor=st->epsfactor;
  int howmany=st->howmany;
  FSLEBin *data=st->data;

  x=(int)(series[act]*epsinv)&nmax;
  y=(int)(series[act+delay*(st->dim-1)]*epsinv)&nmax;
  for (i=x-1;i<=x+1;i++) {
    i1=i&nmax;
    for (j=y-1;j<=y+1;j++) {
      element=box[i1][j&nmax];
      while (element != -1) {
	if (labs(act-element) > mindist) {
	  for (k=0;k<del1;k+=delay) {
	    dx = fabs(series[act+k]-series[element+k]);
	    if (dx > eps)
	      break;
	  }
	  if (k==del1) {
	    if (dx < mindx) {
	      ok=1;
	      if (dx > 0.0) {
		mindx=dx;
		minelement=element;
	      }
	    }
	  }
	}
	element=list[element];
      }
    }
  }

  if ((minelement != -1) && (mindx < eps)) {
    act += del1-delay+1;
    minelement += del1-delay+1;
    /* act/minelement can land exactly on `length` here (one past the last
       valid index, e.g. whenever mindist == 0) - guard before the first
       series[act]/series[minelement] read below, since the bounds checks
       inside the while loops only protect *subsequent* reads, not this
       one. Without this, the very first condition check of either while
       loop is a heap-buffer-overflow read. */
    if (((unsigned long)act >= length) || ((unsigned long)minelement >= length))
      return ok;
    which=(int)(log(mindx/eps0)/log(epsfactor));
    if (which < 0) {
      while ((dx=fabs(series[act]-series[minelement])) < data[0].eps) {
	act++;
	minelement++;
	if (((unsigned long)act >= length) || ((unsigned long)minelement >= length))
	  return ok;
      }
      mindx=dx;
      which=(int)(log(mindx/eps0)/log(epsfactor));
    }
    for (i=which;i<howmany-1;i++) {
      stime=0;
      while ((dx=fabs(series[act]-series[minelement])) < data[i+1].eps) {
	act++;
	minelement++;
	if (((unsigned long)act >= length) || ((unsigned long)minelement >= length))
	  return ok;
	stime++;
      }
      if (stime > 0) {
	data[i].time += stime;
	data[i].factor += log(dx/mindx);
	data[i].count++;
      }
      mindx=dx;
    }
  }
  return ok;
}

FSLEResult *fsle_compute(const double *series_in, unsigned long length,
			  unsigned int dim, unsigned int delay,
			  unsigned int mindist, double eps0, int epsset,
			  FSLEError *error)
{
  unsigned long i, k;
  long n;
  long maxlength;
  double h, se0_av, se0_var, se_av, se_var;
  double min, interval;
  double epsmax;
  double *series;
  char *done, alldone;
  FSLEState st;
  FSLEResult *result;

  if (error != NULL)
    *error = FSLE_OK;

  /* variance(series_in,length,&se0_av,&se0_var) on the raw series */
  se0_av = se0_var = 0.0;
  for (i = 0; i < length; i++) {
    h = series_in[i];
    se0_av += h;
    se0_var += h * h;
  }
  se0_av /= (double)length;
  se0_var = sqrt(fabs(se0_var / (double)length - se0_av * se0_av));
  if (se0_var == 0.0) {
    if (error != NULL)
      *error = FSLE_ERR_ZERO_VARIANCE;
    return NULL;
  }

  /* rescale_data(series,length,&min,&interval), on a private copy */
  check_alloc(series = (double *)malloc(sizeof(double) * length));
  for (i = 0; i < length; i++)
    series[i] = series_in[i];

  min = interval = series[0];
  for (i = 1; i < length; i++) {
    if (series[i] < min) min = series[i];
    if (series[i] > interval) interval = series[i];
  }
  interval -= min;
  if (interval == 0.0) {
    free(series);
    if (error != NULL)
      *error = FSLE_ERR_ZERO_INTERVAL;
    return NULL;
  }
  for (i = 0; i < length; i++)
    series[i] = (series[i] - min) / interval;

  /* variance(series,length,&se_av,&se_var), on the rescaled series */
  se_av = se_var = 0.0;
  for (i = 0; i < length; i++) {
    h = series[i];
    se_av += h;
    se_var += h * h;
  }
  se_av /= (double)length;
  se_var = sqrt(fabs(se_var / (double)length - se_av * se_av));
  if (se_var == 0.0) {
    free(series);
    if (error != NULL)
      *error = FSLE_ERR_ZERO_VARIANCE;
    return NULL;
  }

  if (epsset) {
    eps0 /= interval;
    epsmax = se0_var;
  }
  else {
    eps0 *= se_var;
    epsmax = se_var;
  }
  if (eps0 >= epsmax) {
    free(series);
    if (error != NULL)
      *error = FSLE_ERR_EPS_TOO_LARGE;
    return NULL;
  }

  st.dim = dim;
  st.delay = delay;
  st.mindist = mindist;
  st.nmax = FSLE_NMAX - 1;
  st.length = length;
  st.series = series;
  st.eps0 = eps0;
  st.epsfactor = sqrt(2.0);

  st.howmany = (int)(log(epsmax / eps0) / log(st.epsfactor)) + 1;
  check_alloc(st.data = (FSLEBin *)malloc(sizeof(FSLEBin) * st.howmany));
  st.eps = eps0 / st.epsfactor;
  for (i = 0; i < (unsigned long)st.howmany; i++) {
    st.data[i].time = st.data[i].factor = 0.0;
    st.data[i].eps = (st.eps *= st.epsfactor);
    st.data[i].count = 0;
  }

  check_alloc(st.list = (long *)malloc(sizeof(long) * length));
  check_alloc(done = (char *)malloc(length));
  for (i = 0; i < length; i++)
    done[i] = 0;

  check_alloc(st.box = (long **)malloc(sizeof(long *) * FSLE_NMAX));
  for (i = 0; i < FSLE_NMAX; i++)
    check_alloc(st.box[i] = (long *)malloc(sizeof(long) * FSLE_NMAX));

  maxlength = length - delay * (dim - 1) - 1 - mindist;
  alldone = 0;
  for (st.eps = eps0; (st.eps <= epsmax) && (!alldone); st.eps *= st.epsfactor) {
    st.epsinv = 1.0 / st.eps;
    put_in_boxes(&st);
    alldone = 1;
    for (n = 0; n <= maxlength; n++) {
      if (!done[n])
	done[n] = make_iterate(&st, n);
      alldone &= done[n];
    }
  }

  for (i = 0; i < FSLE_NMAX; i++)
    free(st.box[i]);
  free(st.box);
  free(st.list);
  free(done);
  free(series);

  check_alloc(result = (FSLEResult *)malloc(sizeof(FSLEResult)));
  result->n = 0;
  for (i = 0; i < (unsigned long)st.howmany; i++)
    if (st.data[i].factor > 0.0)
      result->n++;

  if (result->n > 0) {
    check_alloc(result->eps = (double *)malloc(sizeof(double) * result->n));
    check_alloc(result->lyapunov = (double *)malloc(sizeof(double) * result->n));
    check_alloc(result->count = (long *)malloc(sizeof(long) * result->n));
  }
  else {
    result->eps = NULL;
    result->lyapunov = NULL;
    result->count = NULL;
  }

  k = 0;
  for (i = 0; i < (unsigned long)st.howmany; i++)
    if (st.data[i].factor > 0.0) {
      result->eps[k] = st.data[i].eps * interval;
      result->lyapunov[k] = st.data[i].factor / st.data[i].time;
      result->count[k] = st.data[i].count;
      k++;
    }

  free(st.data);
  return result;
}

void fsle_free(FSLEResult *result)
{
  if (result == NULL)
    return;
  free(result->eps);
  free(result->lyapunov);
  free(result->count);
  free(result);
}
