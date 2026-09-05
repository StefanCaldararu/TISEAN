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
/*Author: Rainer Hegger, Last modified: Feb 6, 2006 */
/*Changes:
  Feb 4, 2006: First version
  Feb 6, 2006: Find and remove bugs (1)
  Feb 11, 2006: Add rand_arb_dist to iterate_***_model
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include "routines/tsa.h"
#include "../include/arima-model.h"

#define WID_STR "Fits an multivariate ARIMA model to the data and gives\
 the coefficients\n\tand the residues (or an iterated model)"

unsigned long length=ULONG_MAX,exclude=0;
unsigned int dim=1,poles=10,ilength,ITER=50;
unsigned int arpoles=0,ipoles=0,mapoles=0;
unsigned int verbosity=1;
char *outfile=NULL,*column=NULL,stdo=1,dimset=0,run_model=0,arimaset=0;
char *infile=NULL;
double **series,convergence=1.0e-3;

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
  fprintf(stderr,"\t-p order of initial AR-Fit [default is %u]\n",poles);
  fprintf(stderr,"\t-P order of AR,I,MA-Fit [default is %u,%u,%u]\n",
	  arpoles,ipoles,mapoles);
  fprintf(stderr,"\t-I # of arima iterations [default is %u]\n",ITER);
  fprintf(stderr,"\t-e accuracy of convergence [default is %lf]\n",convergence);
  fprintf(stderr,"\t-s length of iterated model [default no iteration]\n");
  fprintf(stderr,"\t-o output file name [default is 'datafile'.ari]\n");
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
  if ((out=check_option(argv,argc,'P','3')) != NULL) {
    sscanf(out,"%u,%u,%u",&arpoles,&ipoles,&mapoles);
    if ((arpoles+ipoles+mapoles)>0)
      arimaset=1;
  }
  if ((out=check_option(argv,argc,'I','u')) != NULL)
    sscanf(out,"%u",&ITER);
  if ((out=check_option(argv,argc,'e','f')) != NULL)
    sscanf(out,"%lf",&convergence);
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

int main(int argc,char **argv)
{
  char stdi=0;
  long i,j;
  unsigned int is,id;
  FILE *file;
  double **out;
  double avpm,loglikelihood,aic;
  ARIMAModel *model;
  ARIMAModelError fit_error;

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
      strcat(outfile,".ari");
    }
    else {
      check_alloc(outfile=(char*)calloc((size_t)10,(size_t)1));
      strcpy(outfile,"stdin.ari");
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

  model=arima_model_fit((double *const *)series,length,dim,poles,arpoles,
			 ipoles,mapoles,ITER,convergence,&fit_error);

  for (i=0;i<dim;i++)
    free(series[i]);
  free(series);

  if (model == NULL) {
    if (fit_error == ARIMA_MODEL_ERR_ZERO_VARIANCE)
      fprintf(stderr,"Variance of the data is zero. Exiting!\n\n");
    else
      fprintf(stderr,"It makes no sense to have more poles than data! Exiting\n");
    if (outfile != NULL)
      free(outfile);
    if (infile != NULL)
      free(infile);
    exit(fit_error == ARIMA_MODEL_ERR_ZERO_VARIANCE ?
	 VARIANCE_VAR_EQ_ZERO : AR_MODEL_TOO_MANY_POLES);
  }

  avpm=model->rms_error[0]*model->rms_error[0];
  loglikelihood= -log(model->rms_error[0]);
  for (i=1;i<dim;i++) {
    avpm += model->rms_error[i]*model->rms_error[i];
    loglikelihood -= log(model->rms_error[i]);
  }
  loglikelihood *= ((double)model->length);
  loglikelihood += -((double)model->length)*
    ((1.0+log(2.*M_PI))*dim)/2.0;
  avpm=sqrt(avpm/dim);
  if (model->arimaset)
    aic=2.0*(model->arpoles+model->mapoles)-2.0*loglikelihood;
  else
    aic=2.0*model->poles-2.0*loglikelihood;

  if (stdo) {
    if (model->arimaset) {
      printf("#convergence of residuals in arima fit\n");
      for (i=0;i<model->realiter;i++) {
	printf("#iteration %ld ",i+1);
	for (j=0;j<dim;j++)
	  printf("%e ",model->xdiff[i][j]);
	printf("%e",model->diffcoeff[i]);
	printf("\n");
      }
    }
    printf("#average forcast error= %e\n",avpm);
    printf("#individual forecast errors: ");
     for (i=0;i<dim;i++)
      printf("%e ",model->rms_error[i]);
    printf("\n");
    printf("#Log-Likelihood= %e\t AIC= %e\n",loglikelihood,aic);
    for (i=0;i<model->size;i++) {
      id=model->aindex[0][i];
      is=model->aindex[1][i];
      if (id < dim)
	printf("#x_%u(n-%u) ",id+1,is);
      else
	printf("#e_%u(n-%u) ",id+1-dim,is);
      for (j=0;j<dim;j++)
	printf("%e ",model->coeff[j][i]);
      printf("\n");
    }
    if (!run_model || (verbosity&VER_USR1)) {
      for (i=model->order;i<model->length;i++) {
	if (run_model)
	  printf("#");
	for (j=0;j<dim;j++)
	  if (verbosity&VER_USR2)
	    printf("%e %e ",model->series[j][i]+model->average[j],model->residuals[j][i]);
	  else
	    printf("%e ",model->residuals[j][i]);
	printf("\n");
      }
    }
    if (run_model && (ilength > 0)) {
      out=arima_model_iterate(model,ilength,0x44325);
      for (i=0;i<ilength;i++) {
	for (j=0;j<dim;j++)
	  printf("%e ",out[i][j]);
	printf("\n");
      }
      arima_model_iterate_free(out,ilength);
    }
  }
  else {
    file=fopen(outfile,"w");
    if (verbosity&VER_INPUT)
      fprintf(stderr,"Opened %s for output\n",outfile);
    if (model->arimaset) {
      fprintf(file,"#convergence of residuals in arima fit\n");
      for (i=0;i<model->realiter;i++) {
	fprintf(file,"#iteration %ld ",i+1);
	for (j=0;j<dim;j++)
	  fprintf(file,"%e ",model->xdiff[i][j]);
	fprintf(file,"%e",model->diffcoeff[i]);
	fprintf(file,"\n");
      }
    }
    fprintf(file,"#average forcast error= %e\n",avpm);
    fprintf(file,"#individual forecast errors: ");
    for (i=0;i<dim;i++)
      fprintf(file,"%e ",model->rms_error[i]);
    fprintf(file,"\n");
    fprintf(file,"#Log-Likelihood= %e\t AIC= %e\n",loglikelihood,aic);
    for (i=0;i<model->size;i++) {
      id=model->aindex[0][i];
      is=model->aindex[1][i];
      if (id < dim)
	fprintf(file,"#x_%u(n-%u) ",id+1,is);
      else
	fprintf(file,"#e_%u(n-%u) ",id+1-dim,is);
      for (j=0;j<dim;j++)
	fprintf(file,"%e ",model->coeff[j][i]);
      fprintf(file,"\n");
    }
    if (!run_model || (verbosity&VER_USR1)) {
      for (i=model->order;i<model->length;i++) {
	if (run_model)
	  fprintf(file,"#");
	for (j=0;j<dim;j++)
	  if (verbosity&VER_USR2)
	    fprintf(file,"%e %e ",model->series[j][i]+model->average[j],model->residuals[j][i]);
	  else
	    fprintf(file,"%e ",model->residuals[j][i]);
	fprintf(file,"\n");
      }
    }
    if (run_model && (ilength > 0)) {
      out=arima_model_iterate(model,ilength,0x44325);
      for (i=0;i<ilength;i++) {
	for (j=0;j<dim;j++)
	  fprintf(file,"%e ",out[i][j]);
	fprintf(file,"\n");
      }
      arima_model_iterate_free(out,ilength);
    }
    fclose(file);
  }

  if (outfile != NULL)
    free(outfile);
  if (infile != NULL)
    free(infile);
  arima_model_free(model);

  return 0;
}
