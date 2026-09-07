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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "routines/tsa.h"
#include "../include/henon.h"

#define WID_STR "iterate Henon map"

unsigned long length = 0, ntrans = 10000;
char length_set = 0;
/* xinit/yinit, not x0/y0: y0 collides with the libm Bessel function
   built-in of that name when declared at file scope. */
double a = 1.4, b = 0.3, xinit = .68587, yinit = .65876;
char *outfile = NULL, stdo = 1;
unsigned int verbosity = 1;

void show_options(char *progname)
{
  what_i_do(progname, WID_STR);
  fprintf(stderr, " Usage: %s -l# [options]\n", progname);
  fprintf(stderr, " Options:\n");
  fprintf(stderr, "\t-l number of points x,y (l=0: infinite)\n");
  fprintf(stderr, "\t-A parameter a [default 1.4]\n");
  fprintf(stderr, "\t-B parameter b [default 0.3]\n");
  fprintf(stderr, "\t-X initial x [default 0.68587]\n");
  fprintf(stderr, "\t-Y initial y [default 0.65876]\n");
  fprintf(stderr, "\t-x number of transients discarded [default 10000]\n");
  fprintf(stderr, "\t-o output file name, just -o means henon.dat"
	  " [without -o stdout is used]\n");
  fprintf(stderr, "\t-V verbosity level [default 1]\n\t\t"
	  "0='only panic messages'\n\t\t"
	  "1='+ input/output messages'\n");
  fprintf(stderr, "\t-h show these options\n\n");
  exit(0);
}

void scan_options(int argc, char **argv)
{
  char *out;

  if ((out = check_option(argv, argc, 'l', 'u')) != NULL) {
    sscanf(out, "%lu", &length);
    length_set = 1;
  }
  if ((out = check_option(argv, argc, 'x', 'u')) != NULL)
    sscanf(out, "%lu", &ntrans);
  if ((out = check_option(argv, argc, 'A', 'f')) != NULL)
    sscanf(out, "%lf", &a);
  if ((out = check_option(argv, argc, 'B', 'f')) != NULL)
    sscanf(out, "%lf", &b);
  if ((out = check_option(argv, argc, 'X', 'f')) != NULL)
    sscanf(out, "%lf", &xinit);
  if ((out = check_option(argv, argc, 'Y', 'f')) != NULL)
    sscanf(out, "%lf", &yinit);
  if ((out = check_option(argv, argc, 'V', 'u')) != NULL)
    sscanf(out, "%u", &verbosity);
  if ((out = check_option(argv, argc, 'o', 'o')) != NULL) {
    stdo = 0;
    if (strlen(out) > 0)
      outfile = out;
  }
}

int main(int argc, char **argv)
{
  long n;
  double *series;
  FILE *outf;

  if (scan_help(argc, argv))
    show_options(argv[0]);

  scan_options(argc, argv);
#ifndef OMIT_WHAT_I_DO
  if (verbosity & VER_INPUT)
    what_i_do(argv[0], WID_STR);
#endif

  if (!length_set) {
    fprintf(stderr, "henon: -l is required\n");
    exit(1);
  }

  /* The parameters and initial condition pass through fcan() in the
     Fortran, which returns a single precision real that is then widened
     to double -- so the values actually used are double(float(v)), not
     the literal decimals, regardless of whether they came from a flag or
     from the default. */
  a = (double)(float)a;
  b = (double)(float)b;
  xinit = (double)(float)xinit;
  yinit = (double)(float)yinit;

  if (outfile == NULL) {
    check_alloc(outfile = (char *)calloc((size_t)10, (size_t)1));
    strcpy(outfile, "henon.dat");
  }
  if (!stdo)
    test_outfile(outfile);

  if (stdo)
    outf = stdout;
  else {
    outf = fopen(outfile, "w");
    if (outf == NULL) {
      fprintf(stderr, "Cannot open output file %s\n", outfile);
      exit(32);
    }
  }

  /* Iterate the Henon map. length!=0 (an explicit -l) is handled by the
     reentrant henon_generate() API (source_c/api/henon_api.c), shared
     with the Python bindings. length==0 streams forever and can't be
     expressed as a bounded return, so it keeps its own loop below. */
  if (length != 0) {
    series = henon_generate(a, b, xinit, yinit, length, ntrans);
    for (n = 0; n < (long)length; n++)
      fprintf(outf, "%.9e %.9e\n", series[2 * n], series[2 * n + 1]);
    henon_free(series);
  }
  else {
    double xo = xinit, yo = yinit, xn, yn;
    long nn;

    for (nn = -(long)ntrans;; nn++) {
      xn = 1.0 - a * (xo * xo) + b * yo;
      yn = xo;
      xo = xn;
      yo = yn;

      if (nn < 1)
	continue;
      fprintf(outf, "%.9e %.9e\n", (double)(float)xn, (double)(float)yn);
    }
  }

  if (!stdo && outf != NULL)
    fclose(outf);

  return 0;
}
