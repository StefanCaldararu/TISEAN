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
/*Author: Rainer Hegger. Last modified Sep 5, 2004 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include "routines/tsa.h"
#include "../include/polynomp.h"

#define WID_STR "Fits a polynomial to the data."

char *outfile=NULL,stdo=1;
char *parin=NULL,*infile=NULL;
unsigned long length=ULONG_MAX,insample=ULONG_MAX,exclude=0;
unsigned long plength=UINT_MAX;
unsigned long step=1000;
unsigned int column=1,dim=2,delay=1,down_to=1;
unsigned int **order;
unsigned int verbosity=0xff;
double *series,*param;

void show_options(char *progname)
{
  what_i_do(progname,WID_STR);
  fprintf(stderr,"Usage: %s [Options]\n",progname);
  fprintf(stderr,"Options:\n");
  fprintf(stderr,"Everything not being a valid option will be interpreted"
          " as a possible"
          " datafile.\nIf no datafile is given stdin is read. Just - also"
          " means stdin\n");
  fprintf(stderr,"\t-l # of data to use [default: whole file]\n");
  fprintf(stderr,"\t-x # of lines to ignore [default: %lu]\n",exclude);
  fprintf(stderr,"\t-c column to read [default: %u]\n",column);
  fprintf(stderr,"\t-m embedding dimension [default: %u]\n",dim);
  fprintf(stderr,"\t-d delay [default: %u]\n",delay);
  fprintf(stderr,"\t-n insample data [default: all]\n");
  fprintf(stderr,"\t-L length of forecasted series [default: %lu]\n",step);
  fprintf(stderr,"\t-p name of parameter file [default: parameter.pol]\n");
  fprintf(stderr,"\t-o output file name [default: 'datafile'.pbf]\n");
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
    sscanf(out,"%lu",&length);
  if ((out=check_option(in,n,'x','u')) != NULL)
    sscanf(out,"%lu",&exclude);
  if ((out=check_option(in,n,'c','u')) != NULL)
    sscanf(out,"%u",&column);
  if ((out=check_option(in,n,'m','u')) != NULL)
    sscanf(out,"%u",&dim);
  if ((out=check_option(in,n,'d','u')) != NULL)
    sscanf(out,"%u",&delay);
  if ((out=check_option(in,n,'V','u')) != NULL)
    sscanf(out,"%u",&verbosity);
  if ((out=check_option(in,n,'n','u')) != NULL)
    sscanf(out,"%lu",&insample);
  if ((out=check_option(in,n,'L','u')) != NULL)
    sscanf(out,"%lu",&step);
  if ((out=check_option(in,n,'p','s')) != NULL)
    parin=out;
  if ((out=check_option(in,n,'o','o')) != NULL) {
    stdo=0;
    if (strlen(out) > 0)
      outfile=out;
  }
}

int main(int argc,char **argv)
{
  int i,j;
  char stdi=0;
  double **dummy;
  unsigned int *order_flat;
  PolynompResult *result;
  PolynompError error;
  FILE *file;

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
      sprintf(outfile,"%s.pbf",infile);
    }
    else {
      check_alloc(outfile=(char*)calloc((size_t)10,(size_t)1));
      sprintf(outfile,"stdin.pbf");
    }
  }
  if (!stdo)
    test_outfile(outfile);

  if (parin == NULL) {
    check_alloc(parin=(char*)calloc((size_t)14,(size_t)1));
    sprintf(parin,"parameter.pol");
  }
  file=fopen(parin,"r");
  if (file == NULL) {
    fprintf(stderr,"File %s does not exist. Exiting!\n",parin);
    exit(POLYNOMP__WRONG_PARAMETER_FILE);
  }
  fclose(file);

  dummy=(double**)get_multi_series(parin,&plength,0LU,
				   &dim,"",(char)"1",verbosity);

  check_alloc(order=(unsigned int**)malloc(sizeof(int*)*plength));
  for (i=0;i<plength;i++) {
    check_alloc(order[i]=(unsigned int*)malloc(sizeof(int)*dim));
    for (j=0;j<dim;j++)
      order[i][j]=(unsigned int)dummy[j][i];
  }

  series=(double*)get_series(infile,&length,exclude,column,verbosity);

  check_alloc(order_flat=(unsigned int*)malloc(sizeof(unsigned int)*plength*dim));
  for (i=0;i<plength;i++)
    for (j=0;j<dim;j++)
      order_flat[i*dim+j]=order[i][j];

  result=polynomp_fit(series,length,order_flat,(unsigned int)plength,dim,delay,
		       insample,step,&error);
  free(order_flat);
  if (result == NULL) {
    if (error == POLYNOMP_ERR_ZERO_VARIANCE) {
      fprintf(stderr,"Variance of the data is zero. Exiting!\n\n");
      exit(VARIANCE_VAR_EQ_ZERO);
    }
    else {
      fprintf(stderr,"Singular matrix! Exiting!\n");
      exit(SOLVELE_SINGULAR_MATRIX);
    }
  }
  param=result->param;

  if (stdo) {
    if (verbosity&VER_INPUT)
      fprintf(stderr,"Writing to stdout\n");
    fprintf(stdout,"#FCE: %e %e\n",result->fce_insample,result->fce_outsample);
    for (i=0;i<plength;i++) {
      fprintf(stdout,"# ");
      for (j=0;j<dim;j++)
	fprintf(stdout,"%u ",order[i][j]);
      fprintf(stdout,"%e\n",param[i]);
    }
    fflush(stdout);
  }
  else {
    file=fopen(outfile,"w");
    if (verbosity&VER_INPUT)
      fprintf(stderr,"Opened %s for writing\n",outfile);
    fprintf(file,"#FCE: %e %e\n",result->fce_insample,result->fce_outsample);
    for (i=0;i<plength;i++) {
      fprintf(file,"# ");
      for (j=0;j<dim;j++)
	fprintf(file,"%u ",order[i][j]);
      fprintf(file,"%e\n",param[i]);
    }
    fflush(file);
  }

  for (i=0;i<step;i++) {
    if (!stdo) {
      fprintf(file,"%e\n",result->forecast[i]);
      fflush(file);
    }
    else {
      fprintf(stdout,"%e\n",result->forecast[i]);
      fflush(stdout);
    }
  }

  if (!stdo)
    fclose(file);

  polynomp_free(result);

  return 0;
}
