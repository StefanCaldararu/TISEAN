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
/*Author: Rainer Hegger, last modified: Apr 25, 2002 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include "routines/tsa.h"
#include "../include/lyap_r.h"

#define WID_STR "Estimates the maximal Lyapunov exponent; Rosenstein et al."

#define NMAX 256

char *outfile=NULL;
char *infile=NULL;
char epsset=0;
double *series,*lyap;
long box[NMAX][NMAX],*list;
unsigned int dim=2,delay=1,steps=10,mindist=0;
unsigned int column=1;
unsigned int verbosity=0xff;
const unsigned int nmax=NMAX-1;
unsigned long length=ULONG_MAX,exclude=0;
long *found;
double eps0=1.e-3,eps,epsinv;

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
	  "(data interval)/1000]\n");
  fprintf(stderr,"\t-s # of iterations [default 10]\n");
  fprintf(stderr,"\t-o name of output file [default 'datafile'.ros]\n");
  fprintf(stderr,"\t-V verbosity level [default 3]\n\t\t"
          "0='only panic messages'\n\t\t"
          "1='+ input/output messages'\n\t\t"
          "2='+ give more detailed information about the length scales\n");
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
  if ((out=check_option(argv,n,'r','f')) != NULL) {
    epsset=1;
    sscanf(out,"%lf",&eps0);
  }
  if ((out=check_option(argv,n,'s','u')) != NULL)
    sscanf(out,"%u",&steps);
  if ((out=check_option(argv,n,'V','u')) != NULL)
    sscanf(out,"%u",&verbosity);
  if ((out=check_option(argv,n,'o','o')) != NULL)
    if (strlen(out) > 0)
      outfile=out;
}
      
void print_progress(double peps, long found0, void *user_data)
{
  if (verbosity&VER_USR1)
    fprintf(stderr,"epsilon: %e already found: %ld\n",peps,found0);
}

int main(int argc,char **argv)
{
  char stdi=0;
  int i;
  double min,max;
  FILE *file;
  LyapR *result;

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
      strcat(outfile,".ros");
    }
    else {
      check_alloc(outfile=(char*)calloc((size_t)10,(size_t)1));
      strcpy(outfile,"stdin.ros");
    }
  }
  test_outfile(outfile);

  series=(double*)get_series(infile,&length,exclude,column,verbosity);

  /* Mirrors rescale_data()'s own exit condition, checked here (instead of
     inside lyap_r_compute()) so that, exactly like the original, no output
     file gets created/truncated when the data is degenerate. */
  min=max=series[0];
  for (i=1;i<length;i++) {
    if (series[i] < min) min=series[i];
    if (series[i] > max) max=series[i];
  }
  max -= min;
  if (max == 0.0) {
    fprintf(stderr,"rescale_data: data ranges from %e to %e. It makes\n"
	    "\t\tno sense to continue. Exiting!\n\n",min,min+max);
    exit(RESCALE_DATA_ZERO_INTERVAL);
  }

  file=fopen(outfile,"w");
  if (verbosity&VER_INPUT)
    fprintf(stderr,"Opened %s for writing\n",outfile);

  result=lyap_r_compute(series,length,dim,delay,mindist,steps,eps0,epsset,
			 print_progress,NULL);

  for (i=0;i<=steps;i++)
    if (result->found[i])
      fprintf(file,"%d %e\n",i,result->lyap[i]/result->found[i]/2.0);
  fclose(file);

  lyap_r_free(result);

  return 0;
}
