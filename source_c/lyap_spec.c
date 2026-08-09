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
/*Author: Rainer Hegger, last modified Dec 4, 2005  */
/*Changes:
  7/14/05: Changed borders of the sort routine to speed things up
  11/25/05: Show also absolute forecast errors
  12/04/05: Some more changes in sort
  12/20/05: Change in increase neighborhood size loop
  12/28/05: Found bug in memory allocation (index)
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <string.h>
#include "routines/tsa.h"
#include "../include/lyap_spec.h"

#define WID_STR "Estimates the spectrum of Lyapunov exponents using the\n\t\
method of Sano and Sawada."

#define OUT 10

#define DELAY 1

char epsset=0,stdo=1;
char INVERSE,*outfile=NULL;
char *infile=NULL;
char dimset=0;
char *COLUMNS=NULL;
unsigned long LENGTH=ULONG_MAX,ITERATIONS,exclude=0;
unsigned int EMBED=2,DIMENSION=1/*,DELAY=1*/,MINNEIGHBORS=30;
unsigned int verbosity=0xff;
double EPSSTEP=1.2;

double **series;
double epsmin;

void show_options(char *progname)
{
  what_i_do(progname,WID_STR);
  fprintf(stderr," Usage: %s [options]\n",progname);
  fprintf(stderr," Options:\n");
  fprintf(stderr,"Everything not being a valid option will be interpreted"
          " as a possible"
          " datafile.\nIf no datafile is given stdin is read. Just - also"
          " means stdin\n");
  fprintf(stderr,"\t-l # of datapoints [default is whole file]\n");
  fprintf(stderr,"\t-x # of lines to be ignored [default is 0]\n");
  fprintf(stderr,"\t-c column to read[default 1]\n");
  fprintf(stderr,"\t-m # of components,embedding dimension [default %d,%d]\n",
	  DIMENSION,EMBED);
  //  fprintf(stderr,"\t-d delay  [default %d]\n",DELAY);
  fprintf(stderr,"\t-r epsilon size to start with [default "
  "(data interval)/1000]\n");
  fprintf(stderr,"\t-f factor to increase epsilon [default: 1.2]\n");
  fprintf(stderr,"\t-k # of neighbors to use [default: 30]\n");
  fprintf(stderr,"\t-n # of iterations [default: length]\n");
  fprintf(stderr,"\t-I invert the time series [default: no]\n");
  fprintf(stderr,"\t-o name of output file [default 'datafile'.lyaps]\n");
  fprintf(stderr,"\t-V verbosity level [default: 1]\n\t\t"
          "0='only panic messages'\n\t\t"
          "1='+ input/output messages'\n");
  fprintf(stderr,"\t-h show these options\n");
  fprintf(stderr,"\n");
  exit(0);
}

void scan_options(int n,char **argv)
{
  char *out;

  if ((out=check_option(argv,n,'l','u')) != NULL)
    sscanf(out,"%lu",&LENGTH);
  if ((out=check_option(argv,n,'x','u')) != NULL)
    sscanf(out,"%lu",&exclude);
  if ((out=check_option(argv,n,'c','s')) != NULL)
    COLUMNS=out;
  /*  if ((out=check_option(argv,n,'d','u')) != NULL)
      sscanf(out,"%u",&DELAY);*/
  if ((out=check_option(argv,n,'m','2')) != NULL) {
    sscanf(out,"%u,%u",&DIMENSION,&EMBED);
    dimset=1;
  }
  if ((out=check_option(argv,n,'n','u')) != NULL)
    sscanf(out,"%lu",&ITERATIONS);
  if ((out=check_option(argv,n,'r','f')) != NULL) {
    epsset=1;
    sscanf(out,"%lf",&epsmin);
  }
  if ((out=check_option(argv,n,'f','f')) != NULL)
    sscanf(out,"%lf",&EPSSTEP);
  if ((out=check_option(argv,n,'k','u')) != NULL)
    sscanf(out,"%u",&MINNEIGHBORS);
  if ((out=check_option(argv,n,'V','u')) != NULL)
    sscanf(out,"%u",&verbosity);
  if ((out=check_option(argv,n,'I','n')) != NULL)
    INVERSE=1;
  if ((out=check_option(argv,n,'o','o')) != NULL) {
    stdo=0;
    if (strlen(out) > 0)
      outfile=out;
  }
}

typedef struct {
  FILE *file;
  char stdo;
  time_t lasttime;
} ProgressCtx;

/* Reproduces the original main()'s wall-clock-gated progress line: printed
   at most once every OUT seconds, but always on the final iteration
   (is_last), matching the original's
   "if (((time(&newtime)-lasttime) > OUT) || (i == (start-1)))". */
void print_progress(unsigned long count,const double *exponents,
		    unsigned int n,int is_last,void *user_data)
{
  ProgressCtx *ctx=(ProgressCtx*)user_data;
  time_t newtime;
  unsigned int j;

  if (((time(&newtime)-ctx->lasttime) > OUT) || is_last) {
    ctx->lasttime=newtime;
    if (!ctx->stdo) {
      fprintf(ctx->file,"%lu ",count);
      for (j=0;j<n;j++)
	fprintf(ctx->file,"%e ",exponents[j]);
      fprintf(ctx->file,"\n");
      fflush(ctx->file);
    }
    else {
      fprintf(stdout,"%lu ",count);
      for (j=0;j<n;j++)
	fprintf(stdout,"%e ",exponents[j]);
      fprintf(stdout,"\n");
    }
  }
}

int main(int argc,char **argv)
{
  char stdi=0;
  double *interval,*min,maxinterval;
  unsigned int i;
  ProgressCtx ctx;
  LyapSpec *result;

  if (scan_help(argc,argv))
    show_options(argv[0]);

  ITERATIONS=ULONG_MAX;

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
      check_alloc(outfile=(char*)calloc(strlen(infile)+7,(size_t)1));
      strcpy(outfile,infile);
      strcat(outfile,".lyaps");
    }
    else {
      check_alloc(outfile=(char*)calloc((size_t)12,(size_t)1));
      strcpy(outfile,"stdin.lyaps");
    }
  }
  if (!stdo)
    test_outfile(outfile);

  if (COLUMNS == NULL)
    series=(double**)get_multi_series(infile,&LENGTH,exclude,&DIMENSION,"",
				      dimset,verbosity);
  else
    series=(double**)get_multi_series(infile,&LENGTH,exclude,&DIMENSION,
				      COLUMNS,dimset,verbosity);

  if (MINNEIGHBORS > (LENGTH-DELAY*(EMBED-1)-1)) {
    fprintf(stderr,"Your time series is not long enough to find %d neighbors!"
	    " Exiting.\n",MINNEIGHBORS);
    exit(LYAP_SPEC_DATA_TOO_SHORT);
  }

  /* Mirrors rescale_data()'s own exit condition, checked here (instead of
     inside lyap_spec_compute()) so that, exactly like the original, no
     output file gets created/truncated when the data is degenerate, and so
     the exact original message (naming the offending min/max) can still be
     reproduced. */
  check_alloc(min=(double*)malloc(sizeof(double)*DIMENSION));
  check_alloc(interval=(double*)malloc(sizeof(double)*DIMENSION));
  maxinterval=0.0;
  for (i=0;i<DIMENSION;i++) {
    unsigned long k;

    min[i]=interval[i]=series[i][0];
    for (k=1;k<LENGTH;k++) {
      if (series[i][k] < min[i]) min[i]=series[i][k];
      if (series[i][k] > interval[i]) interval[i]=series[i][k];
    }
    interval[i] -= min[i];
    if (interval[i] == 0.0) {
      fprintf(stderr,"rescale_data: data ranges from %e to %e. It makes\n"
	      "\t\tno sense to continue. Exiting!\n\n",min[i],min[i]+interval[i]);
      exit(RESCALE_DATA_ZERO_INTERVAL);
    }
    if (interval[i] > maxinterval)
      maxinterval=interval[i];
  }
  free(min);
  free(interval);

  if (!stdo) {
    ctx.file=fopen(outfile,"w");
    if (verbosity&VER_INPUT)
      fprintf(stderr,"Opened %s for writing\n",outfile);
  }
  else {
    if (verbosity&VER_INPUT)
      fprintf(stderr,"Writing to stdout\n");
  }
  ctx.stdo=stdo;
  time(&ctx.lasttime);

  result=lyap_spec_compute((double *const *)series,LENGTH,DIMENSION,EMBED,
			   ITERATIONS,epsmin,epsset,EPSSTEP,MINNEIGHBORS,
			   INVERSE,print_progress,&ctx);
  if (result == NULL) {
    fprintf(stderr,"#Not enough neighbors found. Exiting\n");
    exit(LYAP_SPEC_NOT_ENOUGH_NEIGHBORS);
  }

  if (!stdo) {
    fprintf(ctx.file,"#Average relative forecast errors:= ");
    for (i=0;i<DIMENSION;i++)
      fprintf(ctx.file,"%e ",result->rel_forecast_error[i]);
    fprintf(ctx.file,"\n");
    fprintf(ctx.file,"#Average absolute forecast errors:= ");
    for (i=0;i<DIMENSION;i++)
      fprintf(ctx.file,"%e ",result->abs_forecast_error[i]);
    fprintf(ctx.file,"\n");
    fprintf(ctx.file,"#Average Neighborhood Size= %e\n",
	    result->avg_neighborhood_size);
    fprintf(ctx.file,"#Average num. of neighbors= %e\n",
	    result->avg_num_neighbors);
    fprintf(ctx.file,"#estimated KY-Dimension= %f\n",result->ky_dimension);
    fclose(ctx.file);
  }
  else {
    fprintf(stdout,"#Average relative forecast errors:= ");
    for (i=0;i<DIMENSION;i++)
      fprintf(stdout,"%e ",result->rel_forecast_error[i]);
    fprintf(stdout,"\n");
    fprintf(stdout,"#Average absolute forecast errors:= ");
    for (i=0;i<DIMENSION;i++)
      fprintf(stdout,"%e ",result->abs_forecast_error[i]);
    fprintf(stdout,"\n");
    fprintf(stdout,"#Average Neighborhood Size= %e\n",
	    result->avg_neighborhood_size);
    fprintf(stdout,"#Average num. of neighbors= %e\n",
	    result->avg_num_neighbors);
    fprintf(stdout,"#estimated KY-Dimension= %f\n",result->ky_dimension);
  }

  lyap_spec_free(result);

  return 0;
}
