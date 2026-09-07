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
/*Author: Rainer Hegger. Last modified: Feb 19, 2007 */
/* Changes:
     2/19/2007:  Changed name and default for noise  
     10/26/2006: Add seed option
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>
#include <math.h>
#include "routines/tsa.h"
#include "../include/lzo-run.h"

#define WID_STR "Makes a local zeroth order forecast for multivariate data\n\
and iterates a trajectory"

char onscreen=1,epsset=0,*outfile=NULL,setsort=1,setnoise=0;
char *infile=NULL;
unsigned int verbosity=0xff;
double **series;

unsigned int embed=2,dim=1,DELAY=1;
char *column=NULL,dimset=0;
unsigned int MINN=50;
unsigned long LENGTH=ULONG_MAX,FLENGTH=1000,exclude=0;
unsigned long seed=0x9074325L;
double EPS0=1.e-3,EPSF=1.2,Q=10.0;

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
  fprintf(stderr,"\t-k # of neighbors  [default %u]\n",MINN);
  fprintf(stderr,"\t-K fix # of neighbors  [default no]\n");
  fprintf(stderr,"\t-%% # variance of noise [default %3.1lf]\n",Q);
  fprintf(stderr,"\t-I seed for the rnd-generator (If seed=0, the time\n"
          "\t\tcommand is used to set the seed) [Default: fixed]\n");
  fprintf(stderr,"\t-r size of initial neighborhood ["
	  " default (data interval)/1000]\n");
  fprintf(stderr,"\t-f factor to increase size [default 1.2]\n");
  fprintf(stderr,"\t-o output file [default 'datafile'.lzr;"
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
  if ((out=check_option(in,n,'K','n')) != NULL)
    setsort=1;
  if ((out=check_option(in,n,'I','u')) != NULL) {
    sscanf(out,"%lu",&seed);
    if (seed == 0)
      seed=(unsigned long)time((time_t*)&seed);
  }
  if ((out=check_option(in,n,'V','u')) != NULL)
    sscanf(out,"%u",&verbosity);
  if ((out=check_option(in,n,'r','f')) != NULL) {
    epsset=1;
    sscanf(out,"%lf",&EPS0);
  }
  if ((out=check_option(in,n,'f','f')) != NULL)
    sscanf(out,"%lf",&EPSF);
  if ((out=check_option(in,n,'%','f')) != NULL) {
    sscanf(out,"%lf",&Q);
    if (Q>0.0)
      setnoise=1;
  }
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
  LzoRun *result;
  LzoRunError error;
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
      check_alloc(outfile=(char*)calloc(strlen(infile)+5,(size_t)1));
      strcpy(outfile,infile);
      strcat(outfile,".lzr");
    }
    else {
      check_alloc(outfile=(char*)calloc((size_t)10,(size_t)1));
      strcpy(outfile,"stdin.lzr");
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

  result=lzo_run_forecast((double *const *)series,LENGTH,dim,embed,DELAY,MINN,
			    setsort,FLENGTH,EPS0,epsset,EPSF,Q,setnoise,seed,
			    &error);
  if (result == NULL) {
    if (error == LZO_RUN_ERR_ZERO_INTERVAL) {
      /* Reproduce rescale_data()'s exact message for whichever component
	 first has zero range. lzo_run_forecast() rescales a private copy
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

  if (!onscreen) {
    file=fopen(outfile,"w");
    if (verbosity&VER_INPUT)
      fprintf(stderr,"Opened %s for writing\n",outfile);
  }
  else {
    if (verbosity&VER_INPUT)
      fprintf(stderr,"Writing to stdout\n");
  }

  for (i=0;i<FLENGTH;i++) {
    if (onscreen) {
      for (j=0;j<dim-1;j++)
	printf("%e ",result->series[i*dim+j]);
      printf("%e\n",result->series[i*dim+dim-1]);
      fflush(stdout);
    }
    else {
      for (j=0;j<dim-1;j++)
	fprintf(file,"%e ",result->series[i*dim+j]);
      fprintf(file,"%e\n",result->series[i*dim+dim-1]);
      fflush(file);
    }
  }
  if (!onscreen)
    fclose(file);

  lzo_run_free(result);
  if (outfile != NULL)
    free(outfile);
  for (i=0;i<dim;i++)
    free(series[i]);
  free(series);

  return 0;
}
