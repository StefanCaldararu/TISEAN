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
/*Author: Rainer Hegger. Last modified: Aug 27, 2004 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "routines/tsa.h"
#include "../include/lzo-test.h"

#define WID_STR "Estimates the average forecast error for a zeroth\n\t\
order fit from a multidimensional time series"


#ifndef _MATH_H
#include <math.h>
#endif

double **series;

char epsset=0,dimset=0,clengthset=0,causalset=0;
char *infile=NULL;
char *outfile=NULL,stdo=1;
char *COLUMNS=NULL;
unsigned int embed=2,dim=1,DELAY=1,MINN=30;
unsigned long STEP=1,causal;
unsigned int verbosity=0x1;
double EPS0=1.e-3,EPSF=1.2;
unsigned long refstep=1;
unsigned long LENGTH=ULONG_MAX,exclude=0,CLENGTH=ULONG_MAX;

void show_options(char *progname)
{
  what_i_do(progname,WID_STR);
  fprintf(stderr," Usage: %s [options]\n",progname);
  fprintf(stderr," Options:\n");
  fprintf(stderr,"Everything not being a valid option will be interpreted"
          " as a possible"
          " datafile.\nIf no datafile is given stdin is read. Just - also"
          " means stdin\n");
  fprintf(stderr,"\t-l # of data to use [default: whole file]\n");
  fprintf(stderr,"\t-x # of lines to be ignored [default: 0]\n");
  fprintf(stderr,"\t-c columns to read [default: 1,...,X]\n");
  fprintf(stderr,"\t-m dimension and embedding dimension"
	  " [default: %d,%d]\n",dim,embed);
  fprintf(stderr,"\t-d delay [default: %d]\n",DELAY);
  fprintf(stderr,"\t-n # of reference points [default: length]\n");
  fprintf(stderr,"\t-S temporal distance between the reference points"
	  " [default: %lu]\n",refstep);
  fprintf(stderr,"\t-k minimal number of neighbors for the fit "
	  "[default: %d]\n",MINN);
  fprintf(stderr,"\t-r neighborhoud size to start with "
	  "[default: (data interval)/1000]\n");
  fprintf(stderr,"\t-f factor to increase size [default: 1.2]\n");
  fprintf(stderr,"\t-s steps to forecast [default: 1]\n");
  fprintf(stderr,"\t-C width of causality window [default: steps]\n");
  fprintf(stderr,"\t-o output file [default: 'datafile.zer',"
	  " without -o: stdout]\n");
  fprintf(stderr,"\t-V verbosity level [default: 1]\n\t\t"
          "0='only panic messages'\n\t\t"
          "1='+ input/output messages'\n\t\t"
	  "2='give individual forecast errors for the max step'\n");
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
  if ((out=check_option(in,n,'c','s')) != NULL)
    COLUMNS=out;
  if ((out=check_option(in,n,'m','2')) != NULL) {
    dimset=1;
    sscanf(out,"%u%*c%u",&dim,&embed);
    if (embed == 0)
      embed=1;
  }
  if ((out=check_option(in,n,'d','u')) != NULL)
    sscanf(out,"%u",&DELAY);
  if ((out=check_option(in,n,'n','u')) != NULL) {
    sscanf(out,"%lu",&CLENGTH);
    clengthset=1;
  }
  if ((out=check_option(in,n,'S','u')) != NULL)
    sscanf(out,"%lu",&refstep);
  if ((out=check_option(in,n,'k','u')) != NULL)
    sscanf(out,"%u",&MINN);
  if ((out=check_option(in,n,'r','f')) != NULL) {
    epsset=1;
    sscanf(out,"%lf",&EPS0);
  }
  if ((out=check_option(in,n,'f','f')) != NULL)
    sscanf(out,"%lf",&EPSF);
  if ((out=check_option(in,n,'s','u')) != NULL)
    sscanf(out,"%lu",&STEP);
  if ((out=check_option(in,n,'C','u')) != NULL) {
    sscanf(out,"%lu",&causal);
    causalset=1;
  }
  if ((out=check_option(in,n,'V','u')) != NULL)
    sscanf(out,"%u",&verbosity);
  if ((out=check_option(in,n,'o','o')) != NULL) {
    stdo=0;
    if (strlen(out) > 0)
      outfile=out;
  }
}

int main(int argc,char **argv)
{
  char stdi=0;
  unsigned long i,j,k;
  unsigned int c;
  double cmin,cinterval;
  LzoTest *result;
  LzoTestError error;
  FILE *file;

  if (scan_help(argc,argv))
    show_options(argv[0]);

  scan_options(argc,argv);

  if ((2*STEP+causal) >= ((long)LENGTH-(long)(embed*DELAY)-(long)MINN)) {
    fprintf(stderr,"steps to forecast (-s) too large. Exiting!\n");
    exit(ZEROTH__STEP_TOO_LARGE);
  }
  if (!causalset)
    causal=STEP;

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
      sprintf(outfile,"%s.zer",infile);
    }
    else {
      check_alloc(outfile=(char*)calloc((size_t)10,(size_t)1));
      sprintf(outfile,"stdin.zer");
    }
  }
  if (!stdo)
    test_outfile(outfile);

  if (COLUMNS == NULL)
    series=(double**)get_multi_series(infile,&LENGTH,exclude,&dim,"",dimset,
				      verbosity);
  else
    series=(double**)get_multi_series(infile,&LENGTH,exclude,&dim,COLUMNS,
				      dimset,verbosity);

  result=lzo_test_compute((double *const *)series,LENGTH,dim,embed,DELAY,MINN,
			   STEP,refstep,causal,CLENGTH,clengthset,EPS0,epsset,
			   EPSF,&error);
  if (result == NULL) {
    if (error == LZO_TEST_ERR_ZERO_INTERVAL) {
      /* Reproduce rescale_data()'s exact message for whichever component
	 first has zero range. lzo_test_compute() rescales a private copy
	 and doesn't report which component failed, but it never touches
	 our own copy of series, so redo the same scan here. */
      for (c=0;c<dim;c++) {
	cmin=cinterval=series[c][0];
	for (k=1;k<LENGTH;k++) {
	  if (series[c][k] < cmin) cmin=series[c][k];
	  if (series[c][k] > cinterval) cinterval=series[c][k];
	}
	cinterval -= cmin;
	if (cinterval == 0.0) break;
      }
      fprintf(stderr,"rescale_data: data ranges from %e to %e. It makes\n"
	      "\t\tno sense to continue. Exiting!\n\n",cmin,cmin+cinterval);
      exit(RESCALE_DATA_ZERO_INTERVAL);
    }
    else {
      fprintf(stderr,"Variance of the data is zero. Exiting!\n\n");
      exit(VARIANCE_VAR_EQ_ZERO);
    }
  }

  if (stdo) {
    if (verbosity&VER_INPUT)
      fprintf(stderr,"Writing to stdout\n");
    for (i=0;i<STEP;i++) {
      if (verbosity&VER_USR1)
	fprintf(stdout,"# %lu ",i+1);
      else
	fprintf(stdout,"%lu ",i+1);
      for (j=0;j<dim;j++)
	fprintf(stdout,"%e ",result->error[i*dim+j]);
      fprintf(stdout,"\n");
    }
    if (verbosity&VER_USR1) {
      for (i=0;i<result->n_ref;i++) {
	for (j=0;j<dim;j++)
	  fprintf(stdout,"%e ",result->diffs[i*dim+j]);
	fprintf(stdout,"\n");
      }
    }
  }
  else {
    file=fopen(outfile,"w");
    if (verbosity&VER_INPUT)
      fprintf(stderr,"Opened %s for writing\n",outfile);
    for (i=0;i<STEP;i++) {
      if (verbosity&VER_USR1)
	fprintf(file,"# %lu ",i+1);
      else
	fprintf(file,"%lu ",i+1);
      for (j=0;j<dim;j++)
	fprintf(file,"%e ",result->error[i*dim+j]);
      fprintf(file,"\n");
    }
    if (verbosity&VER_USR1) {
      for (i=0;i<result->n_ref;i++) {
	for (j=0;j<dim;j++)
	  fprintf(file,"%e ",result->diffs[i*dim+j]);
	fprintf(file,"\n");
      }
    }
    fclose(file);
  }

  lzo_test_free(result);
  if (outfile != NULL)
    free(outfile);
  if (infile != NULL)
    free(infile);
  if (COLUMNS != NULL)
    free(COLUMNS);
  for (i=0;i<dim;i++)
    free(series[i]);
  free(series);

  return 0;
}
