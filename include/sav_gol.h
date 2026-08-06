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

/* Reentrant API for the sav_gol routine: a Savitzky-Golay filter that can
   also estimate filtered derivatives. Extracted out of source_c/sav_gol.c
   so it can be called both from the sav_gol CLI and from other bindings
   (e.g. Python) without going through global state or argv parsing. */

#ifndef _SAV_GOL_H
#define _SAV_GOL_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  unsigned int dim;
  unsigned long length;
  double **data;  /* [dim][length]; filtered series (or estimated
		      derivative, if deriv != 0). The first nb and last nf
		      points per row are left unfiltered (deriv == 0) or
		      set to 0.0 (deriv != 0), matching the CLI's edge
		      handling. */
} SavGol;

/* Applies the Savitzky-Golay filter to series[0..dim-1][0..length-1]: fits
   a degree-`power` polynomial through the `nb` points before and `nf`
   points after each point (except within `nb`/`nf` of either edge) and
   evaluates its `deriv`-th derivative there, normalized by 1/deriv!, the
   same way the sav_gol CLI does it. series is not modified.
   Returns NULL if power >= nb+nf+1 (the fit would be underdetermined) or
   if deriv > power (the derivative order exceeds the polynomial degree). */
SavGol *sav_gol_filter(double *const *series, unsigned long length, unsigned int dim,
			unsigned int nb, unsigned int nf, unsigned int power,
			unsigned int deriv);
void sav_gol_free(SavGol *result);

#ifdef __cplusplus
}
#endif

#endif
