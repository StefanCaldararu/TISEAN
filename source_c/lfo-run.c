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
/*Author: Rainer Hegger. Last modified: Sep 29, 2000 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include "routines/tsa.h"
#include "../include/lfo-run.h"

#define WID_STR "Makes a local linear fit for multivariate data\n\
and iterates a trajectory"

char onscreen=1,epsset=0,*outfile=NULL;
char *infile=NULL;
unsigned int verbosity=0xff;
double **series;

unsigned int embed=2,dim=1,DELAY=1;
char *column=NULL,dimset=0,do_zeroth=0;
int MINN=30;
unsigned long LENGTH=ULONG_MAX,FLENGTH=1000,exclude=0;
double EPS0=1.e-3,EPSF=1.2;

void show_options(char *progname)
{
  what_i_do(progname,WID_STR);
  fprintf(stderr," Usage: %s [Options]\n",progname);
  fprintf(stderr," Options:\n");
  fprintf(stderr,"Everything not being a valid option will be interpreted"
          " as a possible"
          " datafile.\nIf no datafile is given stdin is read. Just - also"
          " means stdin\n");
  fprintf(stderr,"\t-l # of data to be used [default whole file]\n");
  fprintf(stderr,"\t-x # of lines to be ignored [default 0]\n");
  fprintf(stderr,"\t-c column [default 1,...,# of components]\n");
  fprintf(stderr,"\t-m #of components,embedding dimension [default 1,2]\n");
  fprintf(stderr,"\t-d delay for the embedding [default 1]\n");
  fprintf(stderr,"\t-L # of iterations [default 1000]\n");
  fprintf(stderr,"\t-k # of neighbors  [default 30]\n");
  fprintf(stderr,"\t-r size of initial neighborhood ["
	  " default (data interval)/1000]\n");
  fprintf(stderr,"\t-f factor to increase size [default 1.2]\n");
  fprintf(stderr,"\t-0 perfom a zeroth order fit [default not set]\n");
  fprintf(stderr,"\t-o output file [default 'datafile'.cast;"
	  " no -o means write to stdout]\n");
  fprintf(stderr,"\t-V verbosity level [default 1]\n\t\t"
          "0='only panic messages'\n\t\t"
          "1='+ input/output messages'\n");
  fprintf(stderr,"\t-h  show these options\n");
  exit(0);
}

void scan_options(int n,char **in)
{
  char *out;

  if ((out=check_option(in,n,'l','u')) != NULL)
    sscanf(out,"%lu",&LENGTH);
  if ((out=check_option(in,n,'x','u')) != NULL)
    sscanf(out,"%lu",&exclude);
  if ((out=check_option(in,n,'c','s')) != NULL) {
    column=out;
    dimset=1;
  }
  if ((out=check_option(in,n,'m','2')) != NULL)
    sscanf(out,"%u,%u",&dim,&embed);
  if ((out=check_option(in,n,'d','u')) != NULL)
    sscanf(out,"%u",&DELAY);
  if ((out=check_option(in,n,'L','u')) != NULL)
    sscanf(out,"%lu",&FLENGTH);
  if ((out=check_option(in,n,'k','u')) != NULL)
    sscanf(out,"%u",&MINN);
  if ((out=check_option(in,n,'0','n')) != NULL)
    do_zeroth=1;
  if ((out=check_option(in,n,'V','u')) != NULL)
    sscanf(out,"%u",&verbosity);
  if ((out=check_option(in,n,'r','f')) != NULL) {
    epsset=1;
    sscanf(out,"%lf",&EPS0);
  }
  if ((out=check_option(in,n,'f','f')) != NULL)
    sscanf(out,"%lf",&EPSF);
  if ((out=check_option(in,n,'o','o')) != NULL) {
    onscreen=0;
    if (strlen(out) > 0)
      outfile=out;
  }
}

int main(int argc,char **argv)
{
  char stdi=0;
  unsigned int c;
  long i,j;
  unsigned long k;
  double cmin,cinterval;
  LfoRun *result;
  LfoRunError error;
  FILE *file=NULL;

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
      check_alloc(outfile=(char*)calloc(strlen(infile)+6,(size_t)1));
      strcpy(outfile,infile);
      strcat(outfile,".cast");
    }
    else {
      check_alloc(outfile=(char*)calloc((size_t)11,(size_t)1));
      strcpy(outfile,"stdin.cast");
    }
  }
  if (!onscreen)
    test_outfile(outfile);

  if (column == NULL)
    series=(double**)get_multi_series(infile,&LENGTH,exclude,&dim,"",dimset,
				      verbosity);
  else
    series=(double**)get_multi_series(infile,&LENGTH,exclude,&dim,column,
				      dimset,verbosity);

  result=lfo_run_forecast((double *const *)series,LENGTH,dim,embed,DELAY,
			    (unsigned int)MINN,do_zeroth,FLENGTH,EPS0,epsset,
			    EPSF,&error);
  if (result == NULL) {
    /* Reproduce rescale_data()'s exact message for whichever component
       first has zero range. lfo_run_forecast() rescales a private copy and
       doesn't report which component failed, but it never touches our own
       copy of series, so redo the same scan here. */
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

  if (!onscreen) {
    file=fopen(outfile,"w");
    if (verbosity&VER_INPUT)
      fprintf(stderr,"Opened %s for writing\n",outfile);
  }
  else {
    if (verbosity&VER_INPUT)
      fprintf(stderr,"Writing to stdout\n");
  }

  for (i=0;i<(long)result->length;i++) {
    if (onscreen) {
      for (j=0;j<(long)dim-1;j++)
	printf("%e ",result->series[i*dim+j]);
      printf("%e\n",result->series[i*dim+dim-1]);
      fflush(stdout);
    }
    else {
      for (j=0;j<(long)dim-1;j++)
	fprintf(file,"%e ",result->series[i*dim+j]);
      fprintf(file,"%e\n",result->series[i*dim+dim-1]);
      fflush(file);
    }
  }
  if (!onscreen)
    fclose(file);

  if (error == LFO_RUN_ERR_ESCAPED_REGION) {
    fprintf(stderr,"Forecast failed. Escaping data region!\n");
    exit(NSTEP__ESCAPE_REGION);
  }

  lfo_run_free(result);
  if (outfile != NULL)
    free(outfile);
  for (i=0;i<dim;i++)
    free(series[i]);
  free(series);

  return 0;
}
