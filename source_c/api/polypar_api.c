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

/* Reentrant core of polypar, factored out of source_c/polypar.c so it has
   no dependency on argv parsing, file-scope globals, or writing straight
   to a FILE*. The recursion here (a term's exponents are valid as soon as
   their running sum stays <= order) is unchanged from the original
   make_parameter(); only the "sink" for each finished combination changed,
   from an fprintf into the CLI's output file to an append into a growable
   in-memory buffer. */

#include <stdio.h>
#include <stdlib.h>
#include "../routines/tsa.h"
#include "../../include/polypar.h"

typedef struct {
  unsigned int *rows;
  unsigned long count;
  unsigned long capacity;
  unsigned int dim;
} ParamBuffer;

static void param_buffer_push(ParamBuffer *buf, const unsigned int *par)
{
  unsigned int j;

  if (buf->count == buf->capacity) {
    buf->capacity = (buf->capacity == 0) ? 64 : buf->capacity * 2;
    check_alloc(buf->rows = (unsigned int *)realloc(
	buf->rows, sizeof(unsigned int) * buf->capacity * buf->dim));
  }
  for (j = 0; j < buf->dim; j++)
    buf->rows[buf->count * buf->dim + j] = par[j];
  buf->count++;
}

static void make_parameter(ParamBuffer *buf, unsigned int order,
			    unsigned int *par, unsigned int d,
			    unsigned int sum)
{
  int i;

  for (i = 0; i <= (int)order; i++) {
    sum += i;
    if (sum <= order) {
      par[d] = i;
      if (d == 0)
	param_buffer_push(buf, par);
      else
	make_parameter(buf, order, par, d - 1, sum);
    }
    sum -= i;
  }
  par[d] = 0;
}

PolyParResult *polypar_generate(unsigned int dim, unsigned int order)
{
  unsigned int *par;
  ParamBuffer buf;
  PolyParResult *result;

  check_alloc(par = (unsigned int *)malloc(sizeof(unsigned int) * dim));

  buf.rows = NULL;
  buf.count = 0;
  buf.capacity = 0;
  buf.dim = dim;

  make_parameter(&buf, order, par, dim - 1, 0);

  check_alloc(result = (PolyParResult *)malloc(sizeof(PolyParResult)));
  result->dim = dim;
  result->order = order;
  result->count = buf.count;
  result->params = buf.rows;

  free(par);
  return result;
}

void polypar_free(PolyParResult *result)
{
  if (result == NULL)
    return;
  free(result->params);
  free(result);
}
