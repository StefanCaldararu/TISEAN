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
/*Author: Rainer Hegger. Last modified: Sep 4, 1999 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include "routines/tsa.h"
#include "../include/xcor.h"

#define WID_STR "Estimates the crosscorrelations of two data sets\n\t\
given as two columns of one file."

char *columns=NULL,*outfile=NULL,stout=1;
unsigned long length=ULONG_MAX,exclude=0;
long tau=100;
unsigned int verbosity=0xff;
double *array1,*array2;
char *infile=NULL;

void show_options(char *progname) 
{
  what_i_do(progname,WID_STR);
  fprintf(stderr," Usage: %s [Options]\n",progname);
  fprintf(stderr," Options:\n");
  fprintf(stderr,"Everything not being a valid option will be interpreted"
          " as a possible"
          " datafile.\nIf no datafile is given stdin is read. Just - also"
          " means stdin\n");
  fprintf(stderr,"\t-l length [default is whole file]\n");
  fprintf(stderr,"\t-x # of lines to be ignored [default 0]\n");
  fprintf(stderr,"\t-c which columns (separated by commas) [default is 1,2]\n");
  fprintf(stderr,"\t-D corrlength  [default is 100]\n");
  fprintf(stderr,"\t-o output_file  [default is 'datafile'.crc; no -o"
  " means stdout]\n");
  fprintf(stderr,"\t-V verbosity level [default is 1]\n\t\t"
          "0='only panic messages'\n\t\t"
          "1='+ input/output messages'\n");
  fprintf(stderr,"\t-h show these options\n");
  fprintf(stderr,"\n");
  exit(0);
}

void scan_options(int argc,char **argv)
{
  char *out;

  if ((out=check_option(argv,argc,'l','u')) != NULL)
    sscanf(out,"%lu",&length);
  if ((out=check_option(argv,argc,'x','u')) != NULL)
    sscanf(out,"%lu",&exclude);
  if ((out=check_option(argv,argc,'c','s')) != NULL)
    columns=out;
  if ((out=check_option(argv,argc,'D','u')) != NULL)
    sscanf(out,"%ld",&tau);
  if ((out=check_option(argv,argc,'V','u')) != NULL)
    sscanf(out,"%u",&verbosity);
  if ((out=check_option(argv,argc,'o','o')) != NULL) {
    stout=0;
    if (strlen(out) > 0)
      outfile=out;
  }
}

int main(int argc,char** argv)
{
  char stdi=0;
  long i;
  unsigned int dummy=2;
  FILE *fout=NULL;
  double **both;
  XcorResult *result;

  if (scan_help(argc,argv))
    show_options(argv[0]);
  
  scan_options(argc,argv);
#ifndef OMIT_WHAT_I_DO
  if (verbosity&VER_INPUT)
    what_i_do(argv[0],WID_STR);
#endif

  infile=search_datafile(argc,argv,0L,verbosity);
  if (infile == NULL)
    stdi=1;
  
  if (outfile == NULL) {
    if (!stdi) {
      check_alloc(outfile=(char*)calloc(strlen(infile)+5,(size_t)1));
      strcpy(outfile,infile);
      strcat(outfile,".ccr");
    }
    else {
      check_alloc(outfile=(char*)calloc((size_t)10,(size_t)1));
      strcpy(outfile,"stdin.ccr");
    }
  }
  if (!stout)
    test_outfile(outfile);

  if (columns == NULL)
    both=(double**)get_multi_series(infile,&length,exclude,&dummy,"",(char)1,
				    verbosity);
  else
    both=(double**)get_multi_series(infile,&length,exclude,&dummy,columns,
				    (char)1,verbosity);
    
  array1=both[0];
  array2=both[1];

  result=xcor_compute(array1,array2,length,tau);
  if (result == NULL) {
    fprintf(stderr,"Variance of the data is zero. Exiting!\n\n");
    exit(VARIANCE_VAR_EQ_ZERO);
  }

  if (!stout) {
    fout=fopen(outfile,"w");
    if (verbosity&VER_INPUT)
      fprintf(stderr,"Opened %s for writing\n",outfile);
    fprintf(fout,"# average of first comp.=%e\n",result->average1);
    fprintf(fout,"# standard deviation of first comp.=%e\n",result->stddev1);
    fprintf(fout,"# average of sec. comp.=%e\n",result->average2);
    fprintf(fout,"# standard deviation of sec. comp.=%e\n",result->stddev2);
  }
  else {
    if (verbosity&VER_INPUT)
      fprintf(stderr,"Writing to stdout\n");
    fprintf(stdout,"# average of first comp.=%e\n",result->average1);
    fprintf(stdout,"# standard deviation of first comp.=%e\n",result->stddev1);
    fprintf(stdout,"# average of sec. comp.=%e\n",result->average2);
    fprintf(stdout,"# standard deviation of sec. comp.=%e\n",result->stddev2);
  }

  for (i= -(long)result->tau;i<=(long)result->tau;i++)
    if (!stout) {
      fprintf(fout,"%ld %e\n",i,result->values[i+(long)result->tau]);
      fflush(fout);
    }
    else {
      fprintf(stdout,"%ld %e\n",i,result->values[i+(long)result->tau]);
      fflush(stdout);
    }
  if (!stout)
    fclose(fout);

  xcor_free(result);

  return 0;
}
