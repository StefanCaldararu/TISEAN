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

/* Reentrant core of sav_gol, factored out of source_c/sav_gol.c so it has
   no dependency on argv parsing or file-scope globals. The math here
   (make_coeff/make_norm and the three-part filtered/derivative loop) is
   unchanged from the original main(). invert_matrix()/solvele() can still
   exit() the process on a genuinely singular matrix (out-of-memory-style
   internal invariant, not reachable via the power/deriv checks below with
   valid input), the same as every other routine that uses them. */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../routines/tsa.h"
#include "../../include/sav_gol.h"

static double **make_coeff(unsigned int nb, unsigned int nf, unsigned int power)
{
  long i,j,k;
  double **mat,**imat,**rmat;

  check_alloc(mat=(double**)malloc(sizeof(double*)*(power+1)));
  for (i=0;i<=power;i++)
    check_alloc(mat[i]=(double*)malloc(sizeof(double)*(power+1)));
  check_alloc(rmat=(double**)malloc(sizeof(double*)*(power+1)));
  for (i=0;i<=power;i++)
    check_alloc(rmat[i]=(double*)malloc(sizeof(double)*(nb+nf+1)));

  for (i=0;i<=power;i++)
    for (j=0;j<=power;j++) {
      mat[i][j]=0.0;
      for (k= -(int)nb;k<=(int)nf;k++)
	mat[i][j] += pow((double)k,(double)(i+j));
    }

  imat=invert_matrix(mat,(power+1));

  for (i=0;i<=power;i++)
    for (j=0;j<=(nb+nf);j++) {
      rmat[i][j]=0.0;
      for (k=0;k<=power;k++)
	rmat[i][j] += imat[i][k]*pow((double)(j-(int)nb),(double)k);
    }

  for (i=0;i<=power;i++) {
    free(mat[i]);
    free(imat[i]);
  }
  free(mat);
  free(imat);

  return rmat;
}

static double make_norm(unsigned int deriv)
{
  double ret=1.0;
  long i;

  for (i=2;i<=deriv;i++)
    ret *= (double)i;

  return 1.0/ret;
}

SavGol *sav_gol_filter(double *const *series, unsigned long length, unsigned int dim,
			unsigned int nb, unsigned int nf, unsigned int power,
			unsigned int deriv)
{
  long i,j,d;
  double **coeff,help,norm;
  SavGol *result;

  if (power >= (nb+nf+1))
    return NULL;
  if (deriv > power)
    return NULL;

  coeff=make_coeff(nb,nf,power);
  norm=make_norm(deriv);

  check_alloc(result=(SavGol*)malloc(sizeof(SavGol)));
  result->dim=dim;
  result->length=length;
  check_alloc(result->data=(double**)malloc(sizeof(double*)*dim));
  for (d=0;d<dim;d++)
    check_alloc(result->data[d]=(double*)malloc(sizeof(double)*length));

  for (i=0;i<nb;i++)
    for (d=0;d<dim;d++)
      result->data[d][i]=(deriv==0)?series[d][i]:0.0;

  for (i=(long)nb;i<length-(long)nf;i++)
    for (d=0;d<dim;d++) {
      help=0.0;
      for (j= -(long)nb;j<=(long)nf;j++)
	help += coeff[deriv][j+nb]*series[d][i+j];
      result->data[d][i]=help*norm;
    }

  for (i=length-(long)nf;i<length;i++)
    for (d=0;d<dim;d++)
      result->data[d][i]=(deriv==0)?series[d][i]:0.0;

  for (i=0;i<=power;i++)
    free(coeff[i]);
  free(coeff);

  return result;
}

void sav_gol_free(SavGol *result)
{
  unsigned int d;

  if (result == NULL)
    return;
  for (d=0;d<result->dim;d++)
    free(result->data[d]);
  free(result->data);
  free(result);
}
