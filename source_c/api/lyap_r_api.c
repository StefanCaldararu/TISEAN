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

/* Reentrant core of lyap_r, factored out of source_c/lyap_r.c so it has no
   dependency on argv parsing, file-scope globals, or the process-exiting
   error path in the generic rescale_data() library routine it used to
   call. The math here (the box-assisted nearest-neighbor search in
   put_in_boxes()/make_iterate() and the geometric growth of the search
   radius in main()) is unchanged from the original. */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../routines/tsa.h"
#include "../../include/lyap_r.h"

#define LYAP_R_NMAX 256

typedef struct {
  const double *series;
  long **box;
  long *list;
  unsigned int dim, delay, steps, mindist;
  unsigned int nmax;
  unsigned long length;
  double eps, epsinv;
  double *lyap;
  long *found;
} LyapRState;

static void put_in_boxes(LyapRState *st)
{
  int i,j,x,y,del;
  unsigned long length=st->length;
  unsigned int delay=st->delay,dim=st->dim,steps=st->steps,nmax=st->nmax;
  const double *series=st->series;
  long **box=st->box,*list=st->list;
  double epsinv=st->epsinv;

  for (i=0;i<LYAP_R_NMAX;i++)
    for (j=0;j<LYAP_R_NMAX;j++)
      box[i][j]= -1;

  del=delay*(dim-1);
  for (i=0;i<length-del-steps;i++) {
    x=(int)(series[i]*epsinv)&nmax;
    y=(int)(series[i+del]*epsinv)&nmax;
    list[i]=box[x][y];
    box[x][y]=i;
  }
}

static char make_iterate(LyapRState *st, long act)
{
  char ok=0;
  int x,y,i,j,i1,k,del1=st->dim*st->delay;
  long element,minelement= -1;
  double dx,mindx=1.0;
  const double *series=st->series;
  long **box=st->box,*list=st->list;
  unsigned int nmax=st->nmax,mindist=st->mindist,delay=st->delay;
  unsigned int steps=st->steps;
  double eps=st->eps,epsinv=st->epsinv;
  long *found=st->found;
  double *lyap=st->lyap;

  x=(int)(series[act]*epsinv)&nmax;
  y=(int)(series[act+delay*(st->dim-1)]*epsinv)&nmax;
  for (i=x-1;i<=x+1;i++) {
    i1=i&nmax;
    for (j=y-1;j<=y+1;j++) {
      element=box[i1][j&nmax];
      while (element != -1) {
	if (labs(act-element) > mindist) {
	  dx=0.0;
	  for (k=0;k<del1;k+=delay) {
	    dx += (series[act+k]-series[element+k])*
	      (series[act+k]-series[element+k]);
	    if (dx > eps*eps)
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
  if ((minelement != -1) ) {
    act--;
    minelement--;
    for (i=0;i<=steps;i++) {
      act++;
      minelement++;
      dx=0.0;
      for (j=0;j<del1;j+=delay) {
	dx += (series[act+j]-series[minelement+j])*
	  (series[act+j]-series[minelement+j]);
      }
      if (dx > 0.0) {
	found[i]++;
	lyap[i] += log(dx);
      }
    }
  }
  return ok;
}

LyapR *lyap_r_compute(const double *series_in, unsigned long length,
		       unsigned int dim, unsigned int delay,
		       unsigned int mindist, unsigned int steps,
		       double eps0, int epsset,
		       LyapRProgressFn progress, void *user_data)
{
  unsigned long i;
  long n;
  long maxlength;
  double min,max;
  double *series;
  long *list;
  char *done,alldone;
  LyapRState st;
  LyapR *result;

  if (series_in == NULL || length == 0)
    return NULL;

  check_alloc(series=(double*)malloc(sizeof(double)*length));
  for (i=0;i<length;i++)
    series[i]=series_in[i];

  min=max=series[0];
  for (i=1;i<length;i++) {
    if (series[i] < min) min=series[i];
    if (series[i] > max) max=series[i];
  }
  max -= min;
  if (max == 0.0) {
    free(series);
    return NULL;
  }
  for (i=0;i<length;i++)
    series[i]=(series[i]-min)/max;

  if (epsset)
    eps0 /= max;

  check_alloc(list=(long*)malloc(length*sizeof(long)));
  check_alloc(st.lyap=(double*)malloc((steps+1)*sizeof(double)));
  check_alloc(st.found=(long*)malloc((steps+1)*sizeof(long)));
  check_alloc(done=(char*)malloc(length));

  for (i=0;i<=steps;i++) {
    st.lyap[i]=0.0;
    st.found[i]=0;
  }
  for (i=0;i<length;i++)
    done[i]=0;

  check_alloc(st.box=(long**)malloc(sizeof(long*)*LYAP_R_NMAX));
  for (i=0;i<LYAP_R_NMAX;i++)
    check_alloc(st.box[i]=(long*)malloc(sizeof(long)*LYAP_R_NMAX));

  st.series=series;
  st.list=list;
  st.dim=dim;
  st.delay=delay;
  st.steps=steps;
  st.mindist=mindist;
  st.nmax=LYAP_R_NMAX-1;
  st.length=length;

  maxlength=length-delay*(dim-1)-steps-1-mindist;
  alldone=0;
  for (st.eps=eps0;!alldone;st.eps*=1.1) {
    st.epsinv=1.0/st.eps;
    put_in_boxes(&st);
    alldone=1;
    for (n=0;n<=maxlength;n++) {
      if (!done[n])
	done[n]=make_iterate(&st,n);
      alldone &= done[n];
    }
    if (progress != NULL)
      progress(st.eps*max,st.found[0],user_data);
  }

  for (i=0;i<LYAP_R_NMAX;i++)
    free(st.box[i]);
  free(st.box);
  free(list);
  free(done);
  free(series);

  check_alloc(result=(LyapR*)malloc(sizeof(LyapR)));
  result->steps=steps;
  result->found=st.found;
  result->lyap=st.lyap;
  return result;
}

void lyap_r_free(LyapR *result)
{
  if (result == NULL)
    return;
  free(result->found);
  free(result->lyap);
  free(result);
}
