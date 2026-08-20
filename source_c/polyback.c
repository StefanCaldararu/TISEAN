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
/*Author: Rainer Hegger. Last modified Sep 4, 1999 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include "routines/tsa.h"
#include "../include/polyback.h"

#define WID_STR "Does a backward elimination for a polynomial"

char *outfile=NULL,stdo=1;
char *parin=NULL,*infile=NULL;
unsigned long length=ULONG_MAX,insample=ULONG_MAX,exclude=0;
unsigned int column=1,dim=2,delay=1,down_to=1,step=1;
unsigned int verbosity=0xff;
double *series;

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
  fprintf(stderr,"\t-s steps to forecast [default: %u]\n",step);
  fprintf(stderr,"\t-# reduce down to # terms [default: %u]\n",down_to);
  fprintf(stderr,"\t-p name of parameter file [default: parameter.pol]\n");
  fprintf(stderr,"\t-o output file name [default: 'datafile'.pbe]\n");
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
  if ((out=check_option(in,n,'n','u')) != NULL)
    sscanf(out,"%lu",&insample);
  if ((out=check_option(in,n,'#','u')) != NULL)
    sscanf(out,"%u",&down_to);
  if ((out=check_option(in,n,'s','u')) != NULL)
    sscanf(out,"%u",&step);
  if ((out=check_option(in,n,'V','u')) != NULL)
    sscanf(out,"%u",&verbosity);
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
  char stdi=0,*parout;
  double **dummy;
  unsigned long hlength=ULONG_MAX;
  unsigned int **ini_params,*isout,offset,*order_flat;
  FILE *file,*fpars;
  PolybackResult *result;
  PolybackError error;

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
      sprintf(outfile,"%s.pbe",infile);
    }
    else {
      check_alloc(outfile=(char*)calloc((size_t)10,(size_t)1));
      sprintf(outfile,"stdin.pbe");
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
    exit(POLYBACK__WRONG_PARAMETER_FILE);
  }
  fclose(file);

  if (verbosity&VER_INPUT)
    fprintf(stderr,"Using %s as the parameter file\n",parin);
  dummy=(double**)get_multi_series(parin,&hlength,0LU,&dim,"",(char)1,
				   verbosity);

  offset=(unsigned int)(log((double)hlength)/log(10.0)+1.0);
  check_alloc(parout=(char*)calloc(strlen(parin)+offset+2,(size_t)1));

  check_alloc(ini_params=(unsigned int**)malloc(sizeof(int*)*hlength));
  for (i=0;i<hlength;i++) {
    check_alloc(ini_params[i]=(unsigned int*)malloc(sizeof(int)*dim));
    for (j=0;j<dim;j++)
      ini_params[i][j]=(unsigned int)dummy[j][i];
  }
  check_alloc(isout=(unsigned int*)malloc(sizeof(int)*hlength));
  for (i=0;i<hlength;i++)
    isout[i]=0;

  series=(double*)get_series(infile,&length,exclude,column,verbosity);

  check_alloc(order_flat=(unsigned int*)malloc(sizeof(unsigned int)*hlength*dim));
  for (i=0;i<hlength;i++)
    for (j=0;j<dim;j++)
      order_flat[i*dim+j]=ini_params[i][j];

  result=polyback_fit(series,length,order_flat,hlength,dim,delay,insample,
		       step,down_to,&error);
  free(order_flat);
  if (result == NULL) {
    if (error == POLYBACK_ERR_ZERO_VARIANCE) {
      fprintf(stderr,"Variance of the data is zero. Exiting!\n\n");
      exit(VARIANCE_VAR_EQ_ZERO);
    }
    else {
      fprintf(stderr,"Singular matrix! Exiting!\n");
      exit(SOLVELE_SINGULAR_MATRIX);
    }
  }

  if (stdo) {
    fprintf(stdout,"%lu %e %e\n",hlength,result->error_in,result->error_out);
    fflush(stdout);
  }
  else {
    file=fopen(outfile,"w");
    fprintf(file,"%lu %e %e\n",hlength,result->error_in,result->error_out);
    fflush(file);
  }

  for (i=0;i<(int)result->n_levels;i++) {
    unsigned long ibest=result->removed_index[i];

    if (stdo) {
      fprintf(stdout,"%u %e %e ",result->level_n_terms[i],
	      result->level_error_in[i],result->level_error_out[i]);
      for (j=0;j<dim;j++)
	fprintf(stdout,"%u ",ini_params[ibest][j]);
      fprintf(stdout,"\n");
      fflush(stdout);
    }
    else {
      fprintf(file,"%u %e %e ",result->level_n_terms[i],
	      result->level_error_in[i],result->level_error_out[i]);
      for (j=0;j<dim;j++)
	fprintf(file,"%u ",ini_params[ibest][j]);
      fprintf(file,"\n");
      fflush(file);
    }

    isout[ibest]++;
    sprintf(parout,"%s.%u",parin,result->level_n_terms[i]);
    fpars=fopen(parout,"w");
    for (j=0;j<hlength;j++)
      if (!isout[j]) {
	int k;
	for (k=0;k<dim;k++)
	  fprintf(fpars,"%u ",ini_params[j][k]);
	fprintf(fpars,"\n");
      }
    fclose(fpars);
  }

  polyback_free(result);

  if (!stdo)
    fclose(file);
  return 0;
}
