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
/*Author: Rainer Hegger, last modified: Mar 20, 1999 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include "routines/tsa.h"
#include "../include/fsle.h"

#define WID_STR "Estimates the finite size Lyapunov exponent; Vulpiani et al."


char *outfile=NULL;
char *infile=NULL;
char epsset=0,stdo=1;
double *series;
unsigned int dim=2,delay=1,mindist=0;
unsigned int column=1;
unsigned int verbosity=0xff;
unsigned long length=ULONG_MAX,exclude=0;
double eps0=1.e-3;

void show_options(char *progname)
{
  what_i_do(progname,WID_STR);
  fprintf(stderr," Usage: %s [options]\n",progname);
  fprintf(stderr," Options:\n");
  fprintf(stderr,"Everything not being a valid option will be interpreted"
          " as a possible"
          " datafile.\nIf no datafile is given stdin is read. Just - also"
          " means stdin\n");
  fprintf(stderr,"\t-l # of datapoints [default is whole file]\n");
  fprintf(stderr,"\t-x # of lines to be ignored [default is 0]\n");
  fprintf(stderr,"\t-c column to read[default 1]\n");
  fprintf(stderr,"\t-m embedding dimension [default 2]\n");
  fprintf(stderr,"\t-d delay  [default 1]\n");
  fprintf(stderr,"\t-t time window to omit [default 0]\n");
  fprintf(stderr,"\t-r epsilon size to start with [default "
	  "(std. dev. of data)/1000]\n");
  fprintf(stderr,"\t-o name of output file [default 'datafile'.fsl ,"
	  "without -o: stdout]\n");
  fprintf(stderr,"\t-V verbosity level [default 1]\n\t\t"
          "0='only panic messages'\n\t\t"
          "1='+ input/output messages'\n");
  fprintf(stderr,"\t-h show these options\n");
  fprintf(stderr,"\n");
  exit(0);
}

void scan_options(int n,char **argv)
{
  char *out;

  if ((out=check_option(argv,n,'l','u')) != NULL)
    sscanf(out,"%lu",&length);
  if ((out=check_option(argv,n,'x','u')) != NULL)
    sscanf(out,"%lu",&exclude);
  if ((out=check_option(argv,n,'c','u')) != NULL)
    sscanf(out,"%u",&column);
  if ((out=check_option(argv,n,'m','u')) != NULL)
    sscanf(out,"%u",&dim);
  if ((out=check_option(argv,n,'d','u')) != NULL)
    sscanf(out,"%u",&delay);
  if ((out=check_option(argv,n,'t','u')) != NULL)
    sscanf(out,"%u",&mindist);
  if ((out=check_option(argv,n,'V','u')) != NULL)
    sscanf(out,"%u",&verbosity);
  if ((out=check_option(argv,n,'r','f')) != NULL) {
    epsset=1;
    sscanf(out,"%lf",&eps0);
  }
  if ((out=check_option(argv,n,'o','o')) != NULL) {
    stdo=0;
    if (strlen(out) > 0)
      outfile=out;
  }
}

int main(int argc,char **argv)
{
  char stdi=0;
  unsigned long i;
  FILE *file;
  FSLEResult *result;
  FSLEError error;

  if (scan_help(argc,argv))
    show_options(argv[0]);

  scan_options(argc,argv);
#ifndef OMIT_WHAT_I_DO
  if (verbosity&VER_INPUT)
    what_i_do(argv[0],WID_STR);
#endif

  infile=search_datafile(argc,argv,&column,verbosity);
  if (infile == NULL)
    stdi=1;

  if (outfile == NULL) {
    if (!stdi) {
      check_alloc(outfile=(char*)calloc(strlen(infile)+5,(size_t)1));
      strcpy(outfile,infile);
      strcat(outfile,".fsl");
    }
    else {
      check_alloc(outfile=(char*)calloc((size_t)10,(size_t)1));
      strcpy(outfile,"stdin.fsl");
    }
  }
  if (!stdo)
    test_outfile(outfile);

  series=(double*)get_series(infile,&length,exclude,column,verbosity);

  result=fsle_compute(series,length,dim,delay,mindist,eps0,epsset,&error);
  if (result == NULL) {
    if (error == FSLE_ERR_ZERO_VARIANCE) {
      fprintf(stderr,"Variance of the data is zero. Exiting!\n\n");
      exit(VARIANCE_VAR_EQ_ZERO);
    }
    else if (error == FSLE_ERR_ZERO_INTERVAL) {
      /* interval == 0.0 only happens when every point of series equals
	 series[0], so series[0] stands in for both the original
	 rescale_data()'s *min and *min+*interval (which are then also
	 equal). */
      fprintf(stderr,"rescale_data: data ranges from %e to %e. It makes\n"
	      "\t\tno sense to continue. Exiting!\n\n",series[0],series[0]);
      exit(RESCALE_DATA_ZERO_INTERVAL);
    }
    else {
      fprintf(stderr,"The minimal epsilon is too large. Exiting!\n");
      exit(FSLE__TOO_LARGE_MINEPS);
    }
  }

  if (!stdo) {
    file=fopen(outfile,"w");
    if (verbosity&VER_INPUT)
      fprintf(stderr,"Opened %s for writing\n",outfile);
    for (i=0;i<result->n;i++)
      fprintf(file,"%e %e %ld\n",result->eps[i],result->lyapunov[i],result->count[i]);
    fclose(file);
  }
  else {
    if (verbosity&VER_INPUT)
      fprintf(stderr,"Writing to stdout\n");
    for (i=0;i<result->n;i++)
      fprintf(stdout,"%e %e %ld\n",result->eps[i],result->lyapunov[i],result->count[i]);
  }

  fsle_free(result);

  if (infile != NULL)
    free(infile);
  if (outfile != NULL)
    free(outfile);
  free(series);

  return 0;
}
