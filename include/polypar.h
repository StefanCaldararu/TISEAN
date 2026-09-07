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

/* Reentrant API for the polypar routine: enumerates every combination of
   exponents for a multivariate polynomial of a given order. Extracted out
   of source_c/polypar.c so it can be called both from the polypar CLI and
   from other bindings (e.g. Python) without going through global state or
   argv parsing. */

#ifndef _POLYPAR_H
#define _POLYPAR_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  unsigned int dim;
  unsigned int order;
  unsigned long count;   /* number of parameter combinations */
  unsigned int *params;  /* [count][dim] flattened row-major: row i is
			     params[i*dim .. i*dim+dim-1] */
} PolyParResult;

/* Enumerates every combination of dim non-negative integer exponents
   (e_0, ..., e_{dim-1}) with e_0 + ... + e_{dim-1} <= order, in the same
   order source_c/polypar.c's original make_parameter() recursion emitted
   them. dim must be >= 1: like the original recursion this is based on,
   passing dim == 0 is not a supported input (the recursion's depth
   counter is dim - 1, computed as unsigned, so it underflows). */
PolyParResult *polypar_generate(unsigned int dim, unsigned int order);
void polypar_free(PolyParResult *result);

#ifdef __cplusplus
}
#endif

#endif
