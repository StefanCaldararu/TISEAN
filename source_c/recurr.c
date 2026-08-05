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
/*Author: Rainer Hegger Last modified:  Sep 16, 2004 */
/* Sep 16, 2004: Change of index in output. before 0->N-1 now 1->N
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include "routines/tsa.h"
#include "../include/recurr.h"

#define WID_STR "This programs makes a recurrence plot for the data."

unsigned long length=ULONG_MAX,exclude=0;
unsigned int embed=2,dim=1,delay=1;
unsigned int verbosity=0xff;
double eps=1.e-3,fraction=1.0;
char dimset=0;
char *columns;
char *outfile=NULL,stdo=1;
char *infile=NULL;
char epsset=0;

double **series;

void show_options(char *progname)
{
  what_i_do(progname,WID_STR);
  fprintf(stderr,"Usage: %s [options]\n",progname);
  fprintf(stderr,"Options:\n");
  fprintf(stderr,"Everything not being a valid option will be interpreted"
          " as a possible"
          " datafile.\nIf no datafile is given stdin is read. Just - also"
          " means stdin\n");
  fprintf(stderr,"\t-l # of data to use [Default: whole file]\n");
  fprintf(stderr,"\t-x # of lines to be ignored [Default: 0]\n");
  fprintf(stderr,"\t-c columns to read [Default: 1]\n");
  fprintf(stderr,"\t-m # of components,embedding dimension [Default: 1,2]\n");
  fprintf(stderr,"\t-d delay [Default: 1]\n");
  fprintf(stderr,"\t-r size of the neighbourhood "
	  "[Default: (data interval)/1000]\n");
  fprintf(stderr,"\t-%% print only a percentage of points found [Default: "
	  " 100.0]\n");
  fprintf(stderr,"\t-o output file name [Default: 'datafile'.rec\n"
	  "\t\twithout -o: stdout]\n");
  fprintf(stderr,"\t-V verbosity level [Default: 1]\n\t\t"
          "0='only panic messages'\n\t\t"
          "1='+ input/output messages'\n");
  fprintf(stderr,"\t-h show these options\n");
  exit(0);
}

void scan_options(int n,char **in)
{
  char *out;

  if ((out=check_option(in,n,'l','u')) != NULL)
    sscanf(out,"%lu",&length);
  if ((out=check_option(in,n,'x','u')) != NULL)
    sscanf(out,"%lu",&exclude);
  if ((out=check_option(in,n,'c','s')) != NULL)
    columns=out;
  if ((out=check_option(in,n,'m','2')) != NULL) {
    sscanf(out,"%u,%u",&dim,&embed);
    dimset=1;
  }
  if ((out=check_option(in,n,'d','u')) != NULL)
    sscanf(out,"%u",&delay);
  if ((out=check_option(in,n,'V','u')) != NULL)
    sscanf(out,"%u",&verbosity);
  if ((out=check_option(in,n,'r','f')) != NULL) {
    epsset=1;
    sscanf(out,"%lf",&eps);
  }
  if ((out=check_option(in,n,'%','f')) != NULL) {
    sscanf(out,"%lf",&fraction);
    fraction /= 100.0;
  }
  if ((out=check_option(in,n,'o','o')) != NULL) {
    stdo=0;
    if (strlen(out) > 0)
      outfile=out;
  }
}

int main(int argc,char **argv)
{
  unsigned long i;
  char stdi=0;
  double bad_value;
  RecurrResult *result;
  FILE *fout=NULL;

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
      strcpy(outfile,infile);
      strcat(outfile,".rec");
    }
    else {
      check_alloc(outfile=(char*)calloc((size_t)10,(size_t)1));
      strcpy(outfile,"stdin.rec");
    }
  }
  if (!stdo)
    test_outfile(outfile);

  if (columns == NULL)
    series=(double**)get_multi_series(infile,&length,exclude,&dim,"",dimset,
                                      verbosity);
  else
    series=(double**)get_multi_series(infile,&length,exclude,&dim,columns,
                                      dimset,verbosity);

  result=recurr_find((double *const *)series,length,dim,embed,delay,eps,
		      epsset,fraction,&bad_value);
  if (result == NULL) {
    fprintf(stderr,"rescale_data: data ranges from %e to %e. It makes\n"
	    "\t\tno sense to continue. Exiting!\n\n",bad_value,bad_value);
    exit(RESCALE_DATA_ZERO_INTERVAL);
  }

  if (!stdo) {
    fout=fopen(outfile,"w");
    if (verbosity&VER_INPUT)
      fprintf(stderr,"Opened %s for writing\n",outfile);
  }
  else {
    if (verbosity&VER_INPUT)
      fprintf(stderr,"Writing to stdout\n");
  }

  for (i=0;i<result->count;i++) {
    if (!stdo)
      fprintf(fout,"%ld %ld\n",result->point[i],result->neighbor[i]);
    else
      fprintf(stdout,"%ld %ld\n",result->point[i],result->neighbor[i]);
  }
  if (!stdo)
    fclose(fout);

  recurr_free(result);

  return 0;
}
