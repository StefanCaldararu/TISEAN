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
/*Author: Rainer Hegger. Last modified: Mar 11, 2002 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "routines/tsa.h"
#include "../include/rbf.h"
#include <math.h>

#define WID_STR "Fits a RBF-model to the data"

char *outfile=NULL,stdo=1,MAKECAST=0;
char *infile=NULL;
char setdrift=1;
int DIM=2,DELAY=1,CENTER=10,STEP=1;
unsigned int COLUMN=1;
unsigned int verbosity=0xff;
long CLENGTH=1000;
unsigned long LENGTH=ULONG_MAX,INSAMPLE=ULONG_MAX,exclude=0;

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
  fprintf(stderr,"\t-l # of data to use [default: all from file]\n");
  fprintf(stderr,"\t-x # of lines to be ignored [default: 0]\n");
  fprintf(stderr,"\t-c column to read [default: %u]\n",COLUMN);
  fprintf(stderr,"\t-m embedding dimension [default: %d]\n",DIM);
  fprintf(stderr,"\t-d delay [default: %d]\n",DELAY);
  fprintf(stderr,"\t-p number of centers [default: %d]\n",CENTER);
  fprintf(stderr,"\t-X deactivate drift [default: activated]\n");
  fprintf(stderr,"\t-s steps to forecast [default: %d]\n",STEP);
  fprintf(stderr,"\t-n # of points for insample [default: # of data]\n");
  fprintf(stderr,"\t-L steps to cast [default: none]\n");
  fprintf(stderr,"\t-o output file name [default: 'datafile'.rbf]\n");
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
    sscanf(out,"%u",&CENTER);
  if ((out=check_option(in,n,'X','n')) != NULL)
    setdrift=0;
  if ((out=check_option(in,n,'s','u')) != NULL)
    sscanf(out,"%u",&STEP);
  if ((out=check_option(in,n,'V','u')) != NULL)
    sscanf(out,"%u",&verbosity);
  if ((out=check_option(in,n,'n','u')) != NULL)
    sscanf(out,"%lu",&INSAMPLE);
  if ((out=check_option(in,n,'L','u')) != NULL) {
    MAKECAST=1;
    sscanf(out,"%lu",&CLENGTH);
  }
  if ((out=check_option(in,n,'o','o')) != NULL) {
    stdo=0;
    if (strlen(out) > 0)
      outfile=out;
  }
}

int main(int argc,char **argv)
{
  char stdi=0;
  int i,j;
  FILE *file=NULL;
  RBFResult *result;
  RBFError error;

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
      strcat(outfile,".rbf");
    }
    else {
      check_alloc(outfile=(char*)calloc((size_t)10,(size_t)1));
      strcpy(outfile,"stdin.rbf");
    }
  }
  if (!stdo)
    test_outfile(outfile);

  series=(double*)get_series(infile,&LENGTH,exclude,COLUMN,verbosity);

  if (MAKECAST)
    STEP=1;

  result=rbf_fit(series,LENGTH,(unsigned int)DIM,(unsigned int)DELAY,
		  (unsigned int)CENTER,(int)setdrift,(unsigned long)STEP,
		  INSAMPLE,MAKECAST?(unsigned long)CLENGTH:0LU,&error);
  if (result == NULL) {
    if (error == RBF_ERR_ZERO_VARIANCE) {
      fprintf(stderr,"rescale_data: data ranges from %e to %e. It makes\n"
	      "\t\tno sense to continue. Exiting!\n\n",series[0],series[0]);
      exit(RESCALE_DATA_ZERO_INTERVAL);
    }
    else {
      fprintf(stderr,"Singular matrix! Exiting!\n");
      exit(SOLVELE_SINGULAR_MATRIX);
    }
  }

  if (!stdo) {
    file=fopen(outfile,"w");
    if (verbosity&VER_INPUT)
      fprintf(stderr,"Opened %s for writing\n",outfile);
    fprintf(file,"#Center points used:\n");
    for (i=0;i<result->centers;i++) {
      fprintf(file,"#");
      for (j=0;j<DIM;j++)
	fprintf(file," %e",result->center[i][j]);
      fprintf(file,"\n");
    }
    fprintf(file,"#variance= %e\n",result->variance);
    fprintf(file,"#Coefficients:\n");
    fprintf(file,"#%e\n",result->coefs[0]);
    for (i=1;i<=result->centers;i++)
      fprintf(file,"#%e\n",result->coefs[i]);
  }
  else {
    if (verbosity&VER_INPUT)
      fprintf(stderr,"Writing to stdout\n");
    fprintf(stdout,"#Center points used:\n");
    for (i=0;i<result->centers;i++) {
      fprintf(stdout,"#");
      for (j=0;j<DIM;j++)
	fprintf(stdout," %e",result->center[i][j]);
      fprintf(stdout,"\n");
    }
    fprintf(stdout,"#variance= %e\n",result->variance);
    fprintf(stdout,"#Coefficients:\n");
    fprintf(stdout,"#%e\n",result->coefs[0]);
    for (i=1;i<=result->centers;i++)
      fprintf(stdout,"#%e\n",result->coefs[i]);
  }
  if (!stdo)
    fprintf(file,"#insample error= %e\n",result->insample_error);
  else
    fprintf(stdout,"#insample error= %e\n",result->insample_error);

  if (result->has_outsample_error) {
    if (!stdout)
      fprintf(file,"#out of sample error= %e\n",result->outsample_error);
    else
      fprintf(stdout,"#out of sample error= %e\n",result->outsample_error);
  }

  if (MAKECAST) {
    if (!stdo)
      for (i=0;i<result->cast_length;i++)
	fprintf(file,"%e\n",result->cast[i]);
    else
      for (i=0;i<result->cast_length;i++)
	fprintf(stdout,"%e\n",result->cast[i]);
  }

  if (!stdo)
    fclose(file);

  rbf_free(result);

  return 0;
}
