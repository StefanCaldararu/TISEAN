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

/* Reentrant core of lyap_spec, factored out of source_c/lyap_spec.c so it
   has no dependency on argv parsing, file-scope globals, or the
   process-exiting error paths in the generic rescale_data()/variance()
   library routines it used to call (and in its own "not enough neighbors"
   hard-exit). The math here (make_dynamics()'s local-linear fit via a
   box-assisted nearest-neighbor search, sort()'s neighbor selection,
   gram_schmidt()'s reorthonormalization and make_iteration()'s tangent-
   space update) is unchanged from the original, modulo replacing direct
   references to file-scope globals with a per-call state struct. */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include "../routines/tsa.h"
#include "../../include/lyap_spec.h"

#define LYAP_SPEC_BOX 512
#define LYAP_SPEC_EPSMAX 1.0
#define LYAP_SPEC_DELAY 1

typedef struct {
  double **series;          /* [dimension][length], rescaled (and possibly
				time-reversed) private copy */
  unsigned long length;
  unsigned int dimension, embed, alldim;
  unsigned int minneighbors;
  double epsstep;
  double epsmin;             /* mutable: current epsilon threshold */
  int epsset;
  unsigned int **indexes;    /* [2][alldim], from make_multi_index() */
  long **box;                /* [LYAP_SPEC_BOX][LYAP_SPEC_BOX] */
  long *list;                /* [length] */
  unsigned long *found;      /* [length] */
  double *abstand;           /* [length] work buffer for lyap_spec_sort() */
  double **mat;               /* [alldim+1][alldim+1] work buffer */
  double *vec;                 /* [alldim+1] work buffer */
  double *averr;                /* [dimension] accumulator */
  double avneig, aveps;
  unsigned long count;
} LyapSpecState;

static double lyap_spec_sort(LyapSpecState *st, long act,
			      unsigned long *nfound, char *enough)
{
  double maxeps=0.0,dx,dswap,maxdx;
  long self=0,i,j,del,hf,iswap,n1;
  unsigned long imax=*nfound;

  *enough=0;

  for (i=0;i<imax;i++) {
    hf=st->found[i];
    if (hf != act) {
      maxdx=fabs(st->series[0][act]-st->series[0][hf]);
      for (j=1;j<st->alldim;j++) {
	n1=st->indexes[0][j];
	del=st->indexes[1][j];
	dx=fabs(st->series[n1][act-del]-st->series[n1][hf-del]);
	if (dx > maxdx) maxdx=dx;
      }
      st->abstand[i]=maxdx;
    }
    else {
      self=i;
    }
  }

  if (self != (imax-1)) {
    st->abstand[self]=st->abstand[imax-1];
    st->found[self]=st->found[imax-1];
  }

  for (i=0;i<st->minneighbors;i++) {
    for (j=i+1;j<imax-1;j++) {
      if (st->abstand[j]<st->abstand[i]) {
	dswap=st->abstand[i];
	st->abstand[i]=st->abstand[j];
	st->abstand[j]=dswap;
	iswap=st->found[i];
	st->found[i]=st->found[j];
	st->found[j]=iswap;
      }
    }
  }

  if (!st->epsset || (st->abstand[st->minneighbors-1] >= st->epsmin)) {
    *nfound=st->minneighbors;
    *enough=1;
    maxeps=st->abstand[st->minneighbors-1];

    return maxeps;
  }

  for (i=st->minneighbors;i<imax-2;i++) {
    for (j=i+1;j<imax-1;j++) {
      if (st->abstand[j]<st->abstand[i]) {
	dswap=st->abstand[i];
	st->abstand[i]=st->abstand[j];
	st->abstand[j]=dswap;
	iswap=st->found[i];
	st->found[i]=st->found[j];
	st->found[j]=iswap;
      }
    }
    if (st->abstand[i] > st->epsmin) {
      (*nfound)=i+1;
      *enough=1;
      maxeps=st->abstand[i];

      return maxeps;
    }
  }

  maxeps=st->abstand[imax-2];

  return maxeps;
}

/* Returns 0 if fewer than st->minneighbors neighbors were ever found (the
   CLI's "Not enough neighbors found. Exiting" hard-exit case), 1 on
   success. */
static char lyap_spec_make_dynamics(LyapSpecState *st, double **dynamics,
				     long act)
{
  long i,hi,j,hj,k,t=act,d;
  unsigned long nfound=0;
  double **hser,**imat;
  double foundeps=0.0,epsilon,hv,hv1;
  double new_vec;
  char got_enough;
  unsigned int dimension=st->dimension, embed=st->embed, alldim=st->alldim;
  unsigned long length=st->length;

  check_alloc(hser=(double**)malloc(sizeof(double*)*dimension));
  for (i=0;i<dimension;i++)
    hser[i]=st->series[i]+act;

  epsilon=st->epsmin/st->epsstep;
  do {
    epsilon *= st->epsstep;
    if (epsilon > LYAP_SPEC_EPSMAX)
      epsilon=LYAP_SPEC_EPSMAX;
    make_multi_box(st->series,st->box,st->list,length-LYAP_SPEC_DELAY,
		   LYAP_SPEC_BOX,dimension,embed,LYAP_SPEC_DELAY,epsilon);
    nfound=find_multi_neighbors(st->series,st->box,st->list,hser,
				length-LYAP_SPEC_DELAY,LYAP_SPEC_BOX,
				dimension,embed,LYAP_SPEC_DELAY,epsilon,
				st->found);
    if (nfound > st->minneighbors) {
      foundeps=lyap_spec_sort(st,act,&nfound,&got_enough);
      if (got_enough)
	break;
    }
  } while (epsilon < LYAP_SPEC_EPSMAX);

  free(hser);

  st->avneig += nfound;
  st->aveps += foundeps;
  if (!st->epsset)
    st->epsmin=st->aveps/st->count;
  if (nfound < st->minneighbors)
    return 0;

  for (i=0;i<=alldim;i++) {
    st->vec[i]=0.0;
    for (j=0;j<=alldim;j++)
      st->mat[i][j]=0.0;
  }

  for (i=0;i<nfound;i++) {
    act=st->found[i];
    st->mat[0][0] += 1.0;
    for (j=0;j<alldim;j++)
      st->mat[0][j+1] += st->series[st->indexes[0][j]][act-st->indexes[1][j]];
    for (j=0;j<alldim;j++) {
      hv1=st->series[st->indexes[0][j]][act-st->indexes[1][j]];
      hj=j+1;
      for (k=j;k<alldim;k++)
	st->mat[hj][k+1] +=
	  st->series[st->indexes[0][k]][act-st->indexes[1][k]]*hv1;
    }
  }

  for (i=0;i<=alldim;i++)
    for (j=i;j<=alldim;j++)
      st->mat[j][i]=(st->mat[i][j]/=(double)nfound);

  imat=invert_matrix(st->mat,alldim+1);

  for (d=0;d<dimension;d++) {
    for (i=0;i<=alldim;i++)
      st->vec[i]=0.0;
    for (i=0;i<nfound;i++) {
      act=st->found[i];
      hv=st->series[d][act+LYAP_SPEC_DELAY];
      st->vec[0] += hv;
      for (j=0;j<alldim;j++)
	st->vec[j+1] += hv*st->series[st->indexes[0][j]][act-st->indexes[1][j]];
    }
    for (i=0;i<=alldim;i++)
      st->vec[i] /= (double)nfound;

    new_vec=0.0;
    for (i=0;i<=alldim;i++)
      new_vec += imat[0][i]*st->vec[i];
    for (i=1;i<=alldim;i++) {
      hi=i-1;
      dynamics[d][hi]=0.0;
      for (j=0;j<=alldim;j++)
	dynamics[d][hi] += imat[i][j]*st->vec[j];
    }
    for (i=0;i<alldim;i++)
      new_vec += dynamics[d][i]*st->series[st->indexes[0][i]][t-st->indexes[1][i]];
    st->averr[d] += (new_vec-st->series[d][t+LYAP_SPEC_DELAY])*
      (new_vec-st->series[d][t+LYAP_SPEC_DELAY]);
  }

  for (i=0;i<=alldim;i++)
    free(imat[i]);
  free(imat);

  return 1;
}

static void lyap_spec_gram_schmidt(unsigned int alldim, double **delta,
				    double *stretch)
{
  double **dnew,norm,*diff;
  long i,j,k;

  check_alloc(diff=(double*)malloc(sizeof(double)*alldim));
  check_alloc(dnew=(double**)malloc(sizeof(double*)*alldim));
  for (i=0;i<alldim;i++)
    check_alloc(dnew[i]=(double*)malloc(sizeof(double)*alldim));

  for (i=0;i<alldim;i++) {
    for (j=0;j<alldim;j++)
      diff[j]=0.0;
    for (j=0;j<i;j++) {
      norm=0.0;
      for (k=0;k<alldim;k++)
	norm += delta[i][k]*dnew[j][k];
      for (k=0;k<alldim;k++)
	diff[k] -= norm*dnew[j][k];
    }
    norm=0.0;
    for (j=0;j<alldim;j++)
      norm += sqr(delta[i][j]+diff[j]);
    stretch[i]=(norm=sqrt(norm));
    for (j=0;j<alldim;j++)
      dnew[i][j]=(delta[i][j]+diff[j])/norm;
  }
  for (i=0;i<alldim;i++)
    for (j=0;j<alldim;j++)
      delta[i][j]=dnew[i][j];

  free(diff);
  for (i=0;i<alldim;i++)
    free(dnew[i]);
  free(dnew);
}

static void lyap_spec_make_iteration(unsigned int dimension,
				      unsigned int alldim,
				      double **dynamics, double **delta)
{
  double **dnew;
  long i,j,k;

  check_alloc(dnew=(double**)malloc(sizeof(double*)*alldim));
  for (i=0;i<alldim;i++)
    check_alloc(dnew[i]=(double*)malloc(sizeof(double)*alldim));

  for (i=0;i<alldim;i++) {
    for (j=0;j<dimension;j++) {
      dnew[i][j]=dynamics[j][0]*delta[i][0];
      for (k=1;k<alldim;k++)
	dnew[i][j] += dynamics[j][k]*delta[i][k];
    }
    for (j=dimension;j<alldim;j++)
      dnew[i][j]=delta[i][j-1];
  }

  for (i=0;i<alldim;i++)
    for (j=0;j<alldim;j++)
      delta[i][j]=dnew[i][j];

  for (i=0;i<alldim;i++)
    free(dnew[i]);
  free(dnew);
}

LyapSpec *lyap_spec_compute(double *const *series_in, unsigned long length,
			     unsigned int dimension, unsigned int embed,
			     unsigned long iterations,
			     double epsmin, int epsset, double epsstep,
			     unsigned int minneighbors, int inverse,
			     LyapSpecProgressFn progress, void *user_data)
{
  unsigned int alldim = dimension*embed;
  unsigned int i,j;
  unsigned long k;
  char ok = 1;

  double **series=NULL, *hseries=NULL;
  double *minv=NULL, *intervalv=NULL, *av=NULL, *var=NULL, *averr=NULL;
  double maxinterval;
  long **box=NULL;
  long *list=NULL;
  unsigned long *found=NULL;
  double *abstand=NULL;
  double **dynamics=NULL, *factor=NULL, *lfactor=NULL, **delta=NULL;
  double *vec=NULL, **mat=NULL;
  unsigned int **indexes=NULL;
  double *running=NULL;
  unsigned long start, pos;
  LyapSpecState st;
  LyapSpec *result = NULL;

  if (minneighbors > (length - (unsigned long)LYAP_SPEC_DELAY*(embed-1) - 1))
    return NULL;

  check_alloc(series=(double**)malloc(sizeof(double*)*dimension));
  for (i=0;i<dimension;i++) {
    check_alloc(series[i]=(double*)malloc(sizeof(double)*length));
    for (k=0;k<length;k++)
      series[i][k]=series_in[i][k];
  }

  check_alloc(minv=(double*)malloc(sizeof(double)*dimension));
  check_alloc(intervalv=(double*)malloc(sizeof(double)*dimension));
  check_alloc(av=(double*)malloc(sizeof(double)*dimension));
  check_alloc(var=(double*)malloc(sizeof(double)*dimension));
  check_alloc(averr=(double*)malloc(sizeof(double)*dimension));

  maxinterval=0.0;
  for (i=0;i<dimension;i++) {
    double dmin,dinterval,average,variance_val,h;

    averr[i]=0.0;

    dmin=dinterval=series[i][0];
    for (k=1;k<length;k++) {
      if (series[i][k] < dmin) dmin=series[i][k];
      if (series[i][k] > dinterval) dinterval=series[i][k];
    }
    dinterval -= dmin;

    if (dinterval == 0.0) {
      ok=0;
      break;
    }
    if (dinterval > maxinterval)
      maxinterval=dinterval;

    for (k=0;k<length;k++)
      series[i][k]=(series[i][k]-dmin)/dinterval;

    minv[i]=dmin;
    intervalv[i]=dinterval;

    average=variance_val=0.0;
    for (k=0;k<length;k++) {
      h=series[i][k];
      average += h;
      variance_val += h*h;
    }
    average /= (double)length;
    variance_val = sqrt(fabs(variance_val/(double)length-average*average));
    av[i]=average;
    var[i]=variance_val;
  }

  if (!ok)
    goto cleanup;

  if (inverse) {
    check_alloc(hseries=(double*)malloc(sizeof(double)*length));
    for (j=0;j<dimension;j++) {
      for (k=0;k<length;k++)
	hseries[length-1-k]=series[j][k];
      for (k=0;k<length;k++)
	series[j][k]=hseries[k];
    }
    free(hseries);
    hseries=NULL;
  }

  if (!epsset)
    epsmin=1./1000.;
  else
    epsmin /= maxinterval;

  check_alloc(box=(long**)malloc(sizeof(long*)*LYAP_SPEC_BOX));
  for (i=0;i<LYAP_SPEC_BOX;i++)
    check_alloc(box[i]=(long*)malloc(sizeof(long)*LYAP_SPEC_BOX));

  check_alloc(list=(long*)malloc(sizeof(long)*length));
  check_alloc(found=(unsigned long*)malloc(sizeof(long)*length));

  check_alloc(dynamics=(double**)malloc(sizeof(double*)*dimension));
  for (i=0;i<dimension;i++)
    check_alloc(dynamics[i]=(double*)malloc(sizeof(double)*alldim));
  check_alloc(factor=(double*)malloc(sizeof(double)*alldim));
  check_alloc(lfactor=(double*)malloc(sizeof(double)*alldim));
  check_alloc(delta=(double**)malloc(sizeof(double*)*alldim));
  for (i=0;i<alldim;i++)
    check_alloc(delta[i]=(double*)malloc(sizeof(double)*alldim));

  check_alloc(vec=(double*)malloc(sizeof(double)*(alldim+1)));
  check_alloc(mat=(double**)malloc(sizeof(double*)*(alldim+1)));
  for (i=0;i<=alldim;i++)
    check_alloc(mat[i]=(double*)malloc(sizeof(double)*(alldim+1)));

  indexes=(unsigned int**)make_multi_index(dimension,embed,LYAP_SPEC_DELAY);

  rnd_init(0x098342L);
  for (k=0;k<10000;k++)
    rnd_long();
  for (i=0;i<alldim;i++) {
    factor[i]=0.0;
    for (j=0;j<alldim;j++)
      delta[i][j]=(double)rnd_long()/(double)ULONG_MAX;
  }
  lyap_spec_gram_schmidt(alldim,delta,lfactor);

  start=iterations;
  if (start > (length-LYAP_SPEC_DELAY))
    start=length-LYAP_SPEC_DELAY;

  check_alloc(abstand=(double*)malloc(sizeof(double)*length));

  st.series=series;
  st.length=length;
  st.dimension=dimension;
  st.embed=embed;
  st.alldim=alldim;
  st.minneighbors=minneighbors;
  st.epsstep=epsstep;
  st.epsmin=epsmin;
  st.epsset=epsset;
  st.indexes=indexes;
  st.box=box;
  st.list=list;
  st.found=found;
  st.abstand=abstand;
  st.mat=mat;
  st.vec=vec;
  st.averr=averr;
  st.avneig=0.0;
  st.aveps=0.0;
  st.count=0;

  if (progress != NULL)
    check_alloc(running=(double*)malloc(sizeof(double)*alldim));

  for (pos=(unsigned long)(embed-1)*LYAP_SPEC_DELAY;pos<start;pos++) {
    st.count++;
    if (!lyap_spec_make_dynamics(&st,dynamics,(long)pos)) {
      ok=0;
      break;
    }
    lyap_spec_make_iteration(dimension,alldim,dynamics,delta);
    lyap_spec_gram_schmidt(alldim,delta,lfactor);
    for (j=0;j<alldim;j++)
      factor[j] += log(lfactor[j])/(double)LYAP_SPEC_DELAY;

    if (progress != NULL) {
      int is_last = (pos == start-1);
      for (j=0;j<alldim;j++)
	running[j]=factor[j]/(double)st.count;
      progress(st.count,running,alldim,is_last,user_data);
    }
  }

  if (!ok)
    goto cleanup;

  {
    double dimval=0.0;
    unsigned int kyi;

    for (kyi=0;kyi<alldim;kyi++) {
      dimval += factor[kyi];
      if (dimval < 0.0)
	break;
    }
    if (kyi < alldim)
      dimval = kyi + (dimval-factor[kyi])/fabs(factor[kyi]);
    else
      dimval = alldim;

    check_alloc(result=(LyapSpec*)malloc(sizeof(LyapSpec)));
    result->dimension=dimension;
    result->embed=embed;
    result->alldim=alldim;
    result->count=st.count;
    check_alloc(result->exponents=(double*)malloc(sizeof(double)*alldim));
    for (j=0;j<alldim;j++)
      result->exponents[j]=factor[j]/(double)st.count;
    check_alloc(result->rel_forecast_error=
		(double*)malloc(sizeof(double)*dimension));
    check_alloc(result->abs_forecast_error=
		(double*)malloc(sizeof(double)*dimension));
    for (i=0;i<dimension;i++) {
      result->rel_forecast_error[i]=sqrt(averr[i]/(double)st.count)/var[i];
      result->abs_forecast_error[i]=
	sqrt(averr[i]/(double)st.count)*intervalv[i];
    }
    result->avg_neighborhood_size=st.aveps*maxinterval/(double)st.count;
    result->avg_num_neighbors=st.avneig/(double)st.count;
    result->ky_dimension=dimval;
  }

cleanup:
  if (series != NULL) {
    for (i=0;i<dimension;i++)
      free(series[i]);
    free(series);
  }
  free(minv);
  free(intervalv);
  free(av);
  free(var);
  free(averr);
  if (box != NULL) {
    for (i=0;i<LYAP_SPEC_BOX;i++)
      free(box[i]);
    free(box);
  }
  free(list);
  free(found);
  if (dynamics != NULL) {
    for (i=0;i<dimension;i++)
      free(dynamics[i]);
    free(dynamics);
  }
  free(factor);
  free(lfactor);
  if (delta != NULL) {
    for (i=0;i<alldim;i++)
      free(delta[i]);
    free(delta);
  }
  free(vec);
  if (mat != NULL) {
    for (i=0;i<=alldim;i++)
      free(mat[i]);
    free(mat);
  }
  if (indexes != NULL) {
    free(indexes[0]);
    free(indexes[1]);
    free(indexes);
  }
  free(abstand);
  free(running);

  return result;
}

void lyap_spec_free(LyapSpec *result)
{
  if (result == NULL)
    return;
  free(result->exponents);
  free(result->rel_forecast_error);
  free(result->abs_forecast_error);
  free(result);
}
