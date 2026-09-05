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
/* Author: Rainer Hegger Last modified: Feb 22, 2006 */
/* Changes: 
   02/22/06: Remove this strange else in start_box that 
             did not compile anyways
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <limits.h>
#include "routines/tsa.h"
#include "../include/boxcount.h"

#define WID_STR "Estimates the Renyi entropy of Qth order\n\t\
using a partition instead of a covering."

unsigned long LENGTH=ULONG_MAX,exclude=0;
unsigned int maxembed=10,dimension=1,DELAY=1,EPSCOUNT=20;
unsigned int verbosity=0xff;
double Q=2.0,EPSMIN=1.e-3,EPSMAX=1.0;
char dimset=0,epsminset=0,epsmaxset=0;
char *outfile=NULL;
char *column=NULL;

double **series;

void show_options(char *progname)
{
  what_i_do(progname,WID_STR);
  fprintf(stderr,"Usage: %s [Options]\n",progname);
  fprintf(stderr,"Options:\n");
  fprintf(stderr,"\t-l # of datapoints [Default: whole file]\n");
  fprintf(stderr,"\t-x # of lines to ignore [Default: %lu]\n",exclude);
  fprintf(stderr,"\t-M # of columns,maximal embedding dimension "
	  "[Default: %u,%u]\n",dimension,maxembed);
  fprintf(stderr,"\t-c columns to read  [Default: 1,...,#of compon.]\n");
  fprintf(stderr,"\t-d delay [Default: %u]\n",DELAY);
  fprintf(stderr,"\t-Q order of the Renyi entropy [Default: %.1f]\n",Q);
  fprintf(stderr,"\t-r minimal epsilon [Default: (data interval)/1000]\n");
  fprintf(stderr,"\t-R maximal epsilon [Default: data interval]\n");
  fprintf(stderr,"\t-# # of epsilons to use [Default: %u]\n",EPSCOUNT);
  fprintf(stderr,"\t-o output file name [Default: 'datafile'.box]\n");
  fprintf(stderr,"\t-V verbosity level [Default: 1]\n\t\t"
          "0='only panic messages'\n\t\t"
          "1='+ input/output messages'\n");
  fprintf(stderr,"\t-h show these options\n\n");
  exit(0);
}

void scan_options(int n,char **in)
{
  char *out;
  
  if ((out=check_option(in,n,'l','u')) != NULL)
    sscanf(out,"%lu",&LENGTH);
  if ((out=check_option(in,n,'c','s')) != NULL)
    column=out;
  if ((out=check_option(in,n,'x','u')) != NULL)
    sscanf(out,"%lu",&exclude);
  if ((out=check_option(in,n,'M','2')) != NULL) {
    sscanf(out,"%u,%u",&dimension,&maxembed);
    dimset=1;
  }
  if ((out=check_option(in,n,'d','u')) != NULL)
    sscanf(out,"%u",&DELAY);
  if ((out=check_option(in,n,'Q','f')) != NULL)
    sscanf(out,"%lf",&Q);
  if ((out=check_option(in,n,'r','f')) != NULL) {
    sscanf(out,"%lf",&EPSMIN);
    epsminset=1;
  }
  if ((out=check_option(in,n,'R','f')) != NULL) {
    sscanf(out,"%lf",&EPSMAX);
    epsmaxset=1;
  }
  if ((out=check_option(in,n,'#','u')) != NULL)
    sscanf(out,"%u",&EPSCOUNT);
  if ((out=check_option(in,n,'V','u')) != NULL)
    sscanf(out,"%u",&verbosity);
  if ((out=check_option(in,n,'o','s')) != NULL)
    outfile=out;
}

int main(int argc,char **argv)
{
  unsigned int i;
  unsigned long j,k;
  char *infile=NULL,stdi=0;
  FILE *fHq;
  BoxCount *bc;
  BoxCountError error;
  double cmin,cinterval;

  if (scan_help(argc,argv))
    show_options(argv[0]);

  scan_options(argc,argv);
#ifndef OMIT_WHAT_I_DO
  if (verbosity&VER_INPUT)
    what_i_do(argv[0],WID_STR);
#endif

  infile=search_datafile(argc,argv,NULL,verbosity);
  if (infile == NULL)
    stdi=1;

  if (outfile == NULL) {
    if (!stdi) {
      check_alloc(outfile=(char*)calloc(strlen(infile)+5,(size_t)1));
      sprintf(outfile,"%s.box",infile);
    }
    else {
      check_alloc(outfile=(char*)calloc((size_t)10,(size_t)1));
      sprintf(outfile,"stdin.box");
    }
  }
  test_outfile(outfile);

  if (column == NULL)
    series=(double**)get_multi_series(infile,&LENGTH,exclude,&dimension,"",
				      dimset,verbosity);
  else
    series=(double**)get_multi_series(infile,&LENGTH,exclude,&dimension,
				      column,dimset,verbosity);

  bc=boxcount_compute((double *const *)series,LENGTH,dimension,maxembed,
		       DELAY,Q,EPSMIN,epsminset,EPSMAX,epsmaxset,EPSCOUNT,
		       &error);
  if (bc == NULL) {
    /* Reproduce rescale_data()'s exact message for whichever component
       first has zero range. boxcount_compute() rescales a private copy and
       doesn't report which component failed, but it never touches our own
       copy of series, so redo the same scan here (as false_nearest.c does
       for the same situation). */
    for (i=0;i<dimension;i++) {
      cmin=cinterval=series[i][0];
      for (j=1;j<LENGTH;j++) {
	if (series[i][j] < cmin) cmin=series[i][j];
	if (series[i][j] > cinterval) cinterval=series[i][j];
      }
      cinterval -= cmin;
      if (cinterval == 0.0) break;
    }
    fprintf(stderr,"rescale_data: data ranges from %e to %e. It makes\n"
	    "\t\tno sense to continue. Exiting!\n\n",cmin,cmin+cinterval);
    exit(RESCALE_DATA_ZERO_INTERVAL);
  }

  fHq=fopen(outfile,"w");
  if (verbosity&VER_INPUT)
    fprintf(stderr,"Opened %s for writing\n",outfile);

  for (i=0;i<maxembed*dimension;i++) {
    fprintf(fHq,"#component = %d embedding = %d\n",bc->which_component[i]+1,
	    bc->which_embed[i]+1);
    for (k=0;k<EPSCOUNT;k++) {
      if (i == 0)
	fprintf(fHq,"%e %e %e\n",bc->eps[k],bc->entropy[k][i],
		bc->entropy[k][i]);
      else
	fprintf(fHq,"%e %e %e\n",bc->eps[k],bc->entropy[k][i],
		bc->entropy[k][i]-bc->entropy[k][i-1]);
    }
    fprintf(fHq,"\n");
  }
  fclose(fHq);

  boxcount_free(bc);

  return 0;
}
