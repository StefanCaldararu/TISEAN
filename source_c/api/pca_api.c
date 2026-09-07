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

/* Reentrant core of pca, factored out of source_c/pca.c so it has no
   dependency on argv parsing, file-scope globals, or the process-exiting
   error path in the generic eigen() library routine it used to call. The
   math here (the embedded covariance matrix build and its
   eigendecomposition/ordering) is unchanged from main()/make_pca()/ordne(). */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../routines/tsa.h"
#include "../../include/pca.h"

/* tred2/tql2 do the actual eigendecomposition work and are defined
   (non-static) in routines/eigen.c, but only eigen() - the wrapper around
   them that fprintf+exit()s on non-convergence - is declared in tsa.h.
   Calling tred2/tql2 directly lets pca_compute() see tql2's convergence
   flag (ierr) itself and return NULL instead of going through eigen()'s
   exit() path. */
extern int tred2(int *nm, int *n, double *a, double *d, double *e, double *z);
extern int tql2(int *nm, int *n, double *d, double *e, double *z, int *ierr);

/* Sorts eig (length dimemb) descending in place and returns, via ord, the
   original index each sorted position came from. Identical to the
   original ordne() in source_c/pca.c, just parameterized on dimemb instead
   of reading it from a global. */
static void ordne(double *eig, int *ord, unsigned int dimemb)
{
  unsigned int i, j;
  int maxi;
  double max;

  for (i = 0; i < dimemb; i++)
    ord[i] = i;

  for (i = 0; i + 1 < dimemb; i++)
    for (j = i + 1; j < dimemb; j++)
      if (eig[i] < eig[j]) {
	max = eig[i];
	eig[i] = eig[j];
	eig[j] = max;
	maxi = ord[i];
	ord[i] = ord[j];
	ord[j] = maxi;
      }
}

PCA *pca_compute(double *const *series, unsigned long length, unsigned int dim,
		  unsigned int emb, unsigned int delay)
{
  unsigned int dimemb, i, j, k, i1, i2, j1, j2;
  int *ord, ierr, nm;
  double *matarray, **mat, *eig, *trans, *off;
  PCA *pca;

  dimemb = dim * emb;

  check_alloc(eig = (double *)malloc(sizeof(double) * dimemb));
  check_alloc(matarray = (double *)malloc(sizeof(double) * dimemb * dimemb));
  check_alloc(mat = (double **)malloc(sizeof(double *) * dimemb));
  for (i = 0; i < dimemb; i++)
    mat[i] = matarray + i * dimemb;

  for (i = 0; i < dimemb; i++) {
    i1 = i / emb;
    i2 = (i % emb) * delay;
    for (j = i; j < dimemb; j++) {
      j1 = j / emb;
      j2 = (j % emb) * delay;
      mat[i][j] = 0.0;
      for (k = (emb - 1) * delay; k < length; k++)
	mat[i][j] += series[i1][k - i2] * series[j1][k - j2];
      mat[j][i] = (mat[i][j] /= (double)(length - (emb - 1) * delay));
    }
  }

  nm = (int)dimemb;
  check_alloc(trans = (double *)malloc(sizeof(double) * nm * nm));
  check_alloc(off = (double *)malloc(sizeof(double) * nm));
  tred2(&nm, &nm, &mat[0][0], eig, off, trans);
  tql2(&nm, &nm, eig, off, trans, &ierr);

  if (ierr != 0) {
    free(trans);
    free(off);
    free(eig);
    free(matarray);
    free(mat);
    return NULL;
  }

  for (i = 0; i < dimemb; i++)
    for (j = 0; j < dimemb; j++)
      mat[i][j] = trans[i + nm * j];
  free(trans);
  free(off);

  check_alloc(ord = (int *)malloc(sizeof(int) * dimemb));
  ordne(eig, ord, dimemb);

  check_alloc(pca = (PCA *)malloc(sizeof(PCA)));
  pca->dimemb = dimemb;
  pca->eigenvalues = eig;
  check_alloc(pca->eigenvectors = (double **)malloc(sizeof(double *) * dimemb));
  for (i = 0; i < dimemb; i++) {
    check_alloc(pca->eigenvectors[i] = (double *)malloc(sizeof(double) * dimemb));
    for (j = 0; j < dimemb; j++)
      pca->eigenvectors[i][j] = mat[i][ord[j]];
  }

  free(ord);
  free(matarray);
  free(mat);

  return pca;
}

void pca_free(PCA *pca)
{
  unsigned int i;

  if (pca == NULL)
    return;
  for (i = 0; i < pca->dimemb; i++)
    free(pca->eigenvectors[i]);
  free(pca->eigenvectors);
  free(pca->eigenvalues);
  free(pca);
}
