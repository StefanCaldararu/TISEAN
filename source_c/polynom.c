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
/*Author: Rainer Hegger*/
/* Changes:
   6/30/2006: Norm of the errors was wrong
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include "routines/tsa.h"
#include "../include/polynom.h"

#define WID_STR "Fits a polynomial to the data"

char CAST=0,sinsample=0,*outfile=NULL;
char *infile=NULL;
unsigned long LENGTH=ULONG_MAX,exclude=0;
long CLENGTH=1000;
unsigned long INSAMPLE=ULONG_MAX;
int DIM=2,DELAY=1,N=2;
unsigned int COLUMN=1;
unsigned int verbosity=0xff;

double *series;

void show_options(char *progname)
{
  what_i_do(progname,WID_STR);
  fprintf(stderr," Usage: %s [options]\n",progname);
  fprintf(stderr," Options:\n");
  fprintf(stderr,"Everything not being a valid option will be interpreted"
          " as a possible"
          " datafile.\nIf no datafile is given stdin is read. Just - also"
          " means stdin\n");
  fprintf(stderr,"\t-l # of data to use [default: whole file]\n");
  fprintf(stderr,"\t-x # of lines to be ignored [default: 0]\n");
  fprintf(stderr,"\t-c column to read [default: 1]\n");
  fprintf(stderr,"\t-m embedding dimension [default: 2]\n");
  fprintf(stderr,"\t-d delay [default: 1]\n");
  fprintf(stderr,"\t-p order of the polynomial [default: 2]\n");
  fprintf(stderr,"\t-n # of points for insample [default: # of data]\n");
  fprintf(stderr,"\t-L steps to cast [default: none]\n");
  fprintf(stderr,"\t-o output file name [default: 'datafile'.pol]\n");
  fprintf(stderr,"\t-V verbosity level [default: 1]\n\t\t"
          "0='only panic messages'\n\t\t"
          "1='+ input/output messages'\n");
  fprintf(stderr,"\t-h show these options\n");
  exit(0);
}

void scan_options(int n,char **in)
{
  char *out;

  if ((out=check_option(in,n,'l','u')) != NULL)
    sscanf(out,"%lu",&LENGTH);
  if ((out=check_option(in,n,'x','u')) != NULL)
    sscanf(out,"%lu",&exclude);
  if ((out=check_option(in,n,'c','u')) != NULL)
    sscanf(out,"%u",&COLUMN);
  if ((out=check_option(in,n,'m','u')) != NULL)
    sscanf(out,"%u",&DIM);
  if ((out=check_option(in,n,'d','u')) != NULL)
    sscanf(out,"%u",&DELAY);
  if ((out=check_option(in,n,'p','u')) != NULL)
    sscanf(out,"%u",&N);
  if ((out=check_option(in,n,'V','u')) != NULL)
    sscanf(out,"%u",&verbosity);
  if ((out=check_option(in,n,'n','u')) != NULL) {
    sscanf(out,"%lu",&INSAMPLE);
    sinsample=1;
  }
  if ((out=check_option(in,n,'L','u')) != NULL) {
    CAST=1;
    sscanf(out,"%lu",&CLENGTH);
  }
  if ((out=check_option(in,n,'o','o')) != NULL)
    if (strlen(out) > 0)
      outfile=out;
}

int main(int argc,char **argv)
{
  char stdi=0;
  unsigned int j,k;
  FILE *file;
  PolynomResult *result;
  PolynomError error;

  if (scan_help(argc,argv))
    show_options(argv[0]);

  scan_options(argc,argv);
#ifndef OMIT_WHAT_I_DO
  if (verbosity&VER_INPUT)
    what_i_do(argv[0],WID_STR);
#endif

  infile=search_datafile(argc,argv,&COLUMN,verbosity);
  if (infile == NULL)
    stdi=1;

  if (outfile == NULL) {
    if (!stdi) {
      check_alloc(outfile=(char*)calloc(strlen(infile)+5,(size_t)1));
      strcpy(outfile,infile);
      strcat(outfile,".pol");
    }
    else {
      check_alloc(outfile=(char*)calloc((size_t)10,(size_t)1));
      strcpy(outfile,"stdin.pol");
    }
  }
  test_outfile(outfile);

  series=(double*)get_series(infile,&LENGTH,exclude,COLUMN,verbosity);

  result=polynom_fit(series,LENGTH,(unsigned int)DIM,(unsigned int)DELAY,
		      (unsigned int)N,INSAMPLE,CAST?(unsigned long)CLENGTH:0UL,
		      &error);
  if (result == NULL) {
    fprintf(stderr,"Variance of the data is zero. Exiting!\n\n");
    exit(VARIANCE_VAR_EQ_ZERO);
  }

  file=fopen(outfile,"w");
  if (verbosity&VER_INPUT)
    fprintf(stderr,"Opened %s for writing\n",outfile);
  fprintf(file,"#number of free parameters= %u\n\n",result->plength);
  fflush(file);

  fprintf(file,"#used norm for the fit= %e\n",result->norm);

  for (j=0;j<result->plength;j++) {
    fprintf(file,"#");
    for (k=0;k<result->dim;k++)
      fprintf(file,"%d ",result->exponent[j*result->dim+k]);
    fprintf(file,"%e\n",result->coeff[j]);
  }
  fprintf(file,"\n");

  fprintf(file,"#average insample error= %e\n",result->error_insample);

  if (result->has_outsample)
    fprintf(file,"#average out of sample error= %e\n",result->error_outsample);

  if (CAST)
    for (j=0;j<result->step;j++) {
      fprintf(file,"%e\n",result->forecast[j]);
      fflush(file);
    }
  fclose(file);

  polynom_free(result);

  return 0;
}
