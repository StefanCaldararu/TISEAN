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
/*Author: Rainer Hegger. Last modified, Sep 20, 2000 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include "routines/tsa.h"
#include "../include/mutual.h"

#define WID_STR "Estimates the time delayed mutual information\n\t\
of the data set"


char *file_out=NULL,stout=1;
char *infile=NULL;
unsigned long length=ULONG_MAX,exclude=0;
unsigned int column=1;
unsigned int verbosity=0xff;
long partitions=16,corrlength=20;

void show_options(char *progname)
{
  what_i_do(progname,WID_STR);
  fprintf(stderr," Usage: %s [Options]\n\n",progname);
  fprintf(stderr," Options:\n");
  fprintf(stderr,"Everything not being a valid option will be interpreted"
          " as a possible"
          " datafile.\nIf no datafile is given stdin is read. Just - also"
          " means stdin\n");
  fprintf(stderr,"\t-l # of points to be used [Default is all]\n");
  fprintf(stderr,"\t-x # of lines to be ignored [Default is 0]\n");
  fprintf(stderr,"\t-c column to read  [Default is 1]\n");
  fprintf(stderr,"\t-b # of boxes [Default is 16]\n");
  fprintf(stderr,"\t-D max. time delay [Default is 20]\n");
  fprintf(stderr,"\t-o output file [-o without name means 'datafile'.mut;"
	  "\n\t\tNo -o means write to stdout]\n");
  fprintf(stderr,"\t-V verbosity level [Default is 1]\n\t\t"
          "0='only panic messages'\n\t\t"
          "1='+ input/output messages'\n");
  fprintf(stderr,"\t-h  show these options\n");
  fprintf(stderr,"\n");
  exit(0);
}

void scan_options(int n,char** in)
{
  char *out;

  if ((out=check_option(in,n,'l','u')) != NULL)
    sscanf(out,"%lu",&length);
  if ((out=check_option(in,n,'x','u')) != NULL)
    sscanf(out,"%lu",&exclude);
  if ((out=check_option(in,n,'c','u')) != NULL)
    sscanf(out,"%u",&column);
  if ((out=check_option(in,n,'b','u')) != NULL)
    sscanf(out,"%lu",&partitions);
  if ((out=check_option(in,n,'D','u')) != NULL)
    sscanf(out,"%lu",&corrlength);
  if ((out=check_option(in,n,'V','u')) != NULL)
    sscanf(out,"%u",&verbosity);
  if ((out=check_option(in,n,'o','o')) != NULL) {
    stout=0;
    if (strlen(out) > 0)
      file_out=out;
  }
}

int main(int argc,char** argv)
{
  char stdi=0;
  long tau;
  double *series,min,interval;
  unsigned long i;
  FILE *file;
  MutualResult *result;


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

  if (file_out == NULL) {
    if (!stdi) {
      check_alloc(file_out=(char*)calloc(strlen(infile)+5,(size_t)1));
      strcpy(file_out,infile);
      strcat(file_out,".mut");
    }
    else {
      check_alloc(file_out=(char*)calloc((size_t)10,(size_t)1));
      strcpy(file_out,"stdin.mut");
    }
  }
  if (!stout)
    test_outfile(file_out);

  series=(double*)get_series(infile,&length,exclude,column,verbosity);

  result=mutual_compute(series,length,partitions,corrlength);
  if (result == NULL) {
    min=interval=series[0];
    for (i=1;i<length;i++) {
      if (series[i] < min) min=series[i];
      if (series[i] > interval) interval=series[i];
    }
    interval -= min;
    fprintf(stderr,"rescale_data: data ranges from %e to %e. It makes\n"
	    "\t\tno sense to continue. Exiting!\n\n",min,min+interval);
    exit(RESCALE_DATA_ZERO_INTERVAL);
  }
  free(series);

  if (!stout) {
    file=fopen(file_out,"w");
    if (verbosity&VER_INPUT)
      fprintf(stderr,"Opened %s for writing\n",file_out);
    fprintf(file,"#shannon= %e\n",result->values[0]);
    fprintf(file,"%d %e\n",0,result->values[0]);
    for (tau=1;tau<=result->corrlength;tau++) {
      fprintf(file,"%ld %e\n",tau,result->values[tau]);
      fflush(file);
    }
    fclose(file);
  }
  else {
    if (verbosity&VER_INPUT)
      fprintf(stderr,"Writing to stdout\n");
    fprintf(stdout,"#shannon= %e\n",result->values[0]);
    fprintf(stdout,"%d %e\n",0,result->values[0]);
    for (tau=1;tau<=result->corrlength;tau++) {
      fprintf(stdout,"%ld %e\n",tau,result->values[tau]);
      fflush(stdout);
    }
  }

  mutual_free(result);

  return 0;
}

