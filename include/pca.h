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

/* Reentrant API for the pca routine: computes the eigenvalues/eigenvectors
   of the delay-embedded covariance matrix of a multivariate series.
   Extracted out of source_c/pca.c so it can be called both from the pca
   CLI and from other bindings (e.g. Python) without going through global
   state, argv parsing, or the process-exiting error path in eigen(). */

#ifndef _PCA_H
#define _PCA_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  unsigned int dimemb;     /* dim*emb: size of the covariance matrix and
			       number of eigenvalues/eigenvectors */
  double *eigenvalues;     /* [dimemb], sorted descending */
  double **eigenvectors;   /* [dimemb][dimemb]; eigenvectors[i][j] is
			       component i of the eigenvector belonging to
			       eigenvalues[j] (already reordered to match
			       eigenvalues, mirroring the original ordne()) */
} PCA;

/* series is [dim][length] and is expected to already be centered (e.g. via
   variance()/mean subtraction) the same way the pca CLI does it. Builds the
   dim*emb by dim*emb covariance matrix of series, embedding each component
   with emb copies delayed by multiples of delay (matching the CLI's -m/-d
   options), and computes its eigendecomposition. Returns NULL if the
   eigenvalue solver fails to converge (mirrors eigen()'s
   EIG2_TOO_MANY_ITERATIONS exit, but without exiting the process). */
PCA *pca_compute(double *const *series, unsigned long length, unsigned int dim,
		  unsigned int emb, unsigned int delay);
void pca_free(PCA *pca);

#ifdef __cplusplus
}
#endif

#endif
