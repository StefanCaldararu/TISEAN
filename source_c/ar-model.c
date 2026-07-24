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
/*Changes:
  Jun 24, 2005: Output average error for multivariate data
  Nov 25, 2005: Handle model order = 0
  Jan 31, 2006: Add verbosity 4 to print data+residuals
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include "routines/tsa.h"
#include "../include/ar-model.h"

#define WID_STR "Fits an multivariate AR model to the data and gives\
 the coefficients\n\tand the residues (or an iterated model)"

unsigned long length=ULONG_MAX,exclude=0;
unsigned int dim=1,poles=1,ilength;
unsigned int verbosity=1;
char *outfile=NULL,*column=NULL,stdo=1,dimset=0,run_model=0;
char *infile=NULL;
double **series,*my_average;

void show_options(char *progname)
{
  what_i_do(progname,WID_STR);
  fprintf(stderr," Usage: %s [options]\n",progname);
  fprintf(stderr," Options:\n");
  fprintf(stderr,"Everything not being a valid option will be interpreted"
	  " as a possible"
	  " datafile.\nIf no datafile is given stdin is read. Just - also"
	  " means stdin\n");
  fprintf(stderr,"\t-l length of file [default is whole file]\n");
  fprintf(stderr,"\t-x # of lines to be ignored [default is 0]\n");
  fprintf(stderr,"\t-m dimension [default is 1]\n");
  fprintf(stderr,"\t-c columns to read [default is 1,...,dimension]\n");
  fprintf(stderr,"\t-p #order of AR-Fit [default is 1]\n");
  fprintf(stderr,"\t-s length of iterated model [default no iteration]\n");
  fprintf(stderr,"\t-o output file name [default is 'datafile'.ar]\n");
  fprintf(stderr,"\t-V verbosity level [default is 1]\n\t\t"
	  "0='only panic messages'\n\t\t"
	  "1='+ input/output messages'\n\t\t"
	  "2='+ print residuals though iterating a model'\n\t\t"
	  "4='+ print original data plus residuals'\n");
  fprintf(stderr,"\t-h show these options\n\n");
  exit(0);
}

void scan_options(int argc,char **argv)
{
  char *out;

  if ((out=check_option(argv,argc,'p','u')) != NULL) {
    sscanf(out,"%u",&poles);
    if (poles < 1) {
      fprintf(stderr,"The order should at least be one!\n");
      exit(127);
    }
  }
  if ((out=check_option(argv,argc,'l','u')) != NULL)
    sscanf(out,"%lu",&length);
  if ((out=check_option(argv,argc,'x','u')) != NULL)
    sscanf(out,"%lu",&exclude);
  if ((out=check_option(argv,argc,'m','u')) != NULL) {
    sscanf(out,"%u",&dim);
    dimset=1;
  }
  if ((out=check_option(argv,argc,'c','u')) != NULL)
    column=out;
  if ((out=check_option(argv,argc,'V','u')) != NULL)
    sscanf(out,"%u",&verbosity);
  if ((out=check_option(argv,argc,'s','u')) != NULL) {
    sscanf(out,"%u",&ilength);
    run_model=1;
  }
  if ((out=check_option(argv,argc,'o','o')) != NULL) {
    stdo=0;
    if (strlen(out) > 0)
      outfile=out;
  }
}

void set_averages_to_zero(void)
{
  double var;
  long i,j;

  for (i=0;i<dim;i++) {
    variance(series[i],length,&my_average[i],&var);
    for (j=0;j<length;j++)
      series[i][j] -= my_average[i];
  }
}

/* Prints an iterated model to stdout (file==NULL) or to file, in the
   original ar-model CLI format. The seed 0x44325 matches the historical
   CLI behaviour of ar_model_iterate() being always seeded the same way. */
void print_iterated_model(const ARModel *model,FILE *file)
{
  double **out;
  long n,d;

  out=ar_model_iterate(model,ilength,0x44325);

  for (n=0;n<ilength;n++) {
    for (d=0;d<dim;d++) {
      if (file != NULL)
	fprintf(file,"%e ",out[n][d]);
      else
	printf("%e ",out[n][d]);
    }
    if (file != NULL)
      fprintf(file,"\n");
    else
      printf("\n");
  }

  ar_model_iterate_free(out,ilength);
}

int main(int argc,char **argv)
{
  char stdi=0;
  long i,j;
  FILE *file;
  double avpm;
  ARModel *model;

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
      check_alloc(outfile=(char*)calloc(strlen(infile)+4,(size_t)1));
      strcpy(outfile,infile);
      strcat(outfile,".ar");
    }
    else {
      check_alloc(outfile=(char*)calloc((size_t)9,(size_t)1));
      strcpy(outfile,"stdin.ar");
    }
  }
  if (!stdo)
    test_outfile(outfile);

  if (column == NULL)
    series=(double**)get_multi_series(infile,&length,exclude,&dim,"",dimset,
				      verbosity);
  else
    series=(double**)get_multi_series(infile,&length,exclude,&dim,column,
				      dimset,verbosity);

  check_alloc(my_average=(double*)malloc(sizeof(double)*dim));
  set_averages_to_zero();

  if (poles >= length) {
    fprintf(stderr,"It makes no sense to have more poles than data! Exiting\n");
    exit(AR_MODEL_TOO_MANY_POLES);
  }

  model=ar_model_fit((double *const *)series,length,dim,poles);

  if (stdo) {
    avpm=model->rms_error[0]*model->rms_error[0];
    for (i=1;i<dim;i++)
      avpm += model->rms_error[i]*model->rms_error[i];
    avpm=sqrt(avpm/dim);
    printf("#average forcast error= %e\n",avpm);
    printf("#individual forecast errors: ");
    for (i=0;i<dim;i++)
      printf("%e ",model->rms_error[i]);
    printf("\n");
    for (i=0;i<dim*poles;i++) {
      printf("# ");
      for (j=0;j<dim;j++)
	printf("%e ",model->coeff[j][i]);
      printf("\n");
    }
    if (!run_model || (verbosity&VER_USR1)) {
      for (i=poles;i<length;i++) {
	if (run_model)
	  printf("#");
	for (j=0;j<dim;j++)
	  if (verbosity&VER_USR2)
	    printf("%e %e ",series[j][i]+my_average[j],model->residuals[j][i]);
	  else
	    printf("%e ",model->residuals[j][i]);
	printf("\n");
      }
    }
    if (run_model && (ilength > 0))
      print_iterated_model(model,NULL);
  }
  else {
    file=fopen(outfile,"w");
    if (verbosity&VER_INPUT)
      fprintf(stderr,"Opened %s for output\n",outfile);
    avpm=model->rms_error[0]*model->rms_error[0];
    for (i=1;i<dim;i++)
      avpm += model->rms_error[i]*model->rms_error[i];
    avpm=sqrt(avpm/dim);
    fprintf(file,"#average forcast error= %e\n",avpm);
    fprintf(file,"#individual forecast errors: ");
    for (i=0;i<dim;i++)
      fprintf(file,"%e ",model->rms_error[i]);
    fprintf(file,"\n");
    for (i=0;i<dim*poles;i++) {
      fprintf(file,"# ");
      for (j=0;j<dim;j++)
	fprintf(file,"%e ",model->coeff[j][i]);
      fprintf(file,"\n");
    }
    if (!run_model || (verbosity&VER_USR1)) {
      for (i=poles;i<length;i++) {
	if (run_model)
	  fprintf(file,"#");
	for (j=0;j<dim;j++)
	  if (verbosity&VER_USR2)
	    fprintf(file,"%e %e ",series[j][i]+my_average[j],model->residuals[j][i]);
	  else
	    fprintf(file,"%e ",model->residuals[j][i]);
	fprintf(file,"\n");
      }
    }
    if (run_model && (ilength > 0))
      print_iterated_model(model,file);
    fclose(file);
  }

  if (outfile != NULL)
    free(outfile);
  if (infile != NULL)
    free(infile);
  ar_model_free(model);

  return 0;
}
