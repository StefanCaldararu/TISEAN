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

/* Reentrant core of recurr, factored out of source_c/recurr.c so it has no
   dependency on argv parsing, file-scope globals, or the process-exiting
   error path in the generic rescale_data() library routine it used to
   call. The math here (per-dimension rescale to [0,1), the box-assisted
   neighbor search and its fraction-based random thinning) is unchanged
   from main()/lfind_neighbors(); the only difference is that found
   neighbor pairs are appended to a growable array instead of being
   printed immediately. */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include "../routines/tsa.h"
#include "../../include/recurr.h"

#define BOX 1024
#define RECURR_CHUNK 1024

RecurrResult *recurr_find(double *const *series, unsigned long length,
			   unsigned int dim, unsigned int embed,
			   unsigned int delay, double eps, char eps_is_raw,
			   double fraction, double *bad_value)
{
  unsigned int d;
  unsigned long ui;
  int i,i1,i2,j,j1,ke,ked,kd;
  int ibox=BOX-1;
  long n,element;
  double dx,epsinv,min,interval,maxinterval;
  double **rescaled;
  long **box,*list;
  char toolarge;
  unsigned long capacity,count;
  long *point,*neighbor;
  RecurrResult *result;

  if (dim == 0 || length == 0)
    return NULL;

  check_alloc(rescaled=(double**)malloc(sizeof(double*)*dim));
  maxinterval=0.0;
  for (d=0;d<dim;d++) {
    check_alloc(rescaled[d]=(double*)malloc(sizeof(double)*length));
    min=interval=series[d][0];
    for (ui=1;ui<length;ui++) {
      if (series[d][ui] < min)
	min=series[d][ui];
      if (series[d][ui] > interval)
	interval=series[d][ui];
    }
    interval -= min;
    if (interval == 0.0) {
      if (bad_value != NULL)
	*bad_value=min;
      for (ui=0;ui<=d;ui++)
	free(rescaled[ui]);
      free(rescaled);
      return NULL;
    }
    for (ui=0;ui<length;ui++)
      rescaled[d][ui]=(series[d][ui]-min)/interval;
    if (interval > maxinterval)
      maxinterval=interval;
  }

  if (eps_is_raw)
    eps /= maxinterval;

  check_alloc(list=(long*)malloc(sizeof(long)*length));
  check_alloc(box=(long**)malloc(sizeof(long*)*BOX));
  for (ui=0;ui<BOX;ui++)
    check_alloc(box[ui]=(long*)malloc(sizeof(long)*BOX));

  make_multi_box(rescaled,box,list,length,BOX,dim,embed,delay,eps);

  epsinv=1./eps;
  rnd_init(0x9834725L);

  capacity=RECURR_CHUNK;
  count=0;
  check_alloc(point=(long*)malloc(sizeof(long)*capacity));
  check_alloc(neighbor=(long*)malloc(sizeof(long)*capacity));

  for (n=(long)((embed-1)*delay);(unsigned long)n<length;n++) {
    i=(int)(rescaled[0][n]*epsinv)&ibox;
    j=(int)(rescaled[dim-1][n]*epsinv)&ibox;
    for (i1=i-1;i1<=i+1;i1++) {
      i2=i1&ibox;
      for (j1=j-1;j1<=j+1;j1++) {
	element=box[i2][j1&ibox];
	while (element > n) {
	  toolarge=0;
	  for (ke=0;ke<(int)embed;ke++) {
	    ked=ke*delay;
	    for (kd=0;kd<(int)dim;kd++) {
	      dx=fabs(rescaled[kd][n-ked]-rescaled[kd][element-ked]);
	      if (dx >= eps) {
		toolarge=1;
		break;
	      }
	    }
	    if (toolarge)
	      break;
	  }
	  if (!toolarge) {
	    if (((double)rnd69069()/ULONG_MAX) <= fraction) {
	      if (count == capacity) {
		capacity += RECURR_CHUNK;
		check_alloc(point=(long*)realloc(point,sizeof(long)*capacity));
		check_alloc(neighbor=(long*)realloc(neighbor,sizeof(long)*capacity));
	      }
	      point[count]=n+1;
	      neighbor[count]=element+1;
	      count++;
	    }
	  }
	  element=list[element];
	}
      }
    }
  }

  for (ui=0;ui<BOX;ui++)
    free(box[ui]);
  free(box);
  free(list);
  for (d=0;d<dim;d++)
    free(rescaled[d]);
  free(rescaled);

  check_alloc(result=(RecurrResult*)malloc(sizeof(RecurrResult)));
  result->count=count;
  result->point=point;
  result->neighbor=neighbor;
  return result;
}

void recurr_free(RecurrResult *result)
{
  if (result == NULL)
    return;
  free(result->point);
  free(result->neighbor);
  free(result);
}
