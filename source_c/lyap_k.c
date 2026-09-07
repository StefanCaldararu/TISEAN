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
/*Author: Rainer Hegger. Last modified: Sep 3, 1999*/
#include <math.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "routines/tsa.h"
#include "../include/lyap_k.h"

#define WID_STR "Estimates the maximal Lyapunov exponent using the Kantz\n\t\
algorithm"

unsigned long length=ULONG_MAX;
unsigned long exclude=0;
unsigned long reference=ULONG_MAX;
unsigned int maxdim=2;
unsigned int mindim=2;
unsigned int delay=1;
unsigned int column=1;
unsigned int epscount=5;
unsigned int maxiter=50;
unsigned int window=0;
unsigned int verbosity=0xff;
double epsmin=1.e-3,epsmax=1.e-2;
char eps0set=0,eps1set=0;
char *outfile=NULL;
char *infile=NULL;

double *series;

void show_options(char *progname)
{
  what_i_do(progname,WID_STR);

  fprintf(stderr," Usage: %s [options]\n",progname);
  fprintf(stderr," Options:\n");
  fprintf(stderr,"Everything not being a valid option will be "
	  "interpreted as a possible datafile.\nIf no datafile "
	  "is given stdin is read. Just - also means stdin\n");
  fprintf(stderr,"\t-l # of data [default: whole file]\n");
  fprintf(stderr,"\t-x # of lines to be ignored [default: 0]\n");
  fprintf(stderr,"\t-c column to read [default: 1]\n");
  fprintf(stderr,"\t-M maxdim [default: 2]\n");
  fprintf(stderr,"\t-m mindim [default: 2]\n");
  fprintf(stderr,"\t-d delay [default: 1]\n");
  fprintf(stderr,"\t-r mineps [default: (data interval)/1000]\n");
  fprintf(stderr,"\t-R maxeps [default: (data interval)/100]\n");
  fprintf(stderr,"\t-# # of eps [default: 5]\n");
  fprintf(stderr,"\t-n # of reference points [default: # of data]\n");
  fprintf(stderr,"\t-s # of iterations [default: 50]\n");
  fprintf(stderr,"\t-t time window [default: 0]\n");
  fprintf(stderr,"\t-o outfile [default: 'datafile'.lyap]\n");
  fprintf(stderr,"\t-V verbosity level [default: 3]\n\t\t"
	  "0='only panic messages'\n\t\t"
	  "1='+ input/output messages'\n\t\t"
	  "2='+ plus statistics'\n");
  fprintf(stderr,"\t-h show these options\n");
  exit(0);
}

void scan_options(int n,char **str)
{
  char *out;
  
  if ((out=check_option(str,n,'l','u')) != NULL)
    sscanf(out,"%lu",&length);
  if ((out=check_option(str,n,'x','u')) != NULL)
    sscanf(out,"%lu",&exclude);
  if ((out=check_option(str,n,'c','u')) != NULL)
    sscanf(out,"%u",&column);
  if ((out=check_option(str,n,'M','u')) != NULL)
    sscanf(out,"%u",&maxdim);
  if ((out=check_option(str,n,'m','u')) != NULL)
    sscanf(out,"%u",&mindim);
  if ((out=check_option(str,n,'d','u')) != NULL)
    sscanf(out,"%u",&delay);
  if ((out=check_option(str,n,'r','f')) != NULL) {
    eps0set=1;
    sscanf(out,"%lf",&epsmin);
  }
  if ((out=check_option(str,n,'R','f')) != NULL) {
    eps1set=1;
    sscanf(out,"%lf",&epsmax);
  }
  if ((out=check_option(str,n,'#','u')) != NULL)
    sscanf(out,"%u",&epscount);
  if ((out=check_option(str,n,'n','u')) != NULL)
    sscanf(out,"%lu",&reference);
  if ((out=check_option(str,n,'s','u')) != NULL)
    sscanf(out,"%u",&maxiter);
  if ((out=check_option(str,n,'t','u')) != NULL)
    sscanf(out,"%u",&window);
  if ((out=check_option(str,n,'V','u')) != NULL)
    sscanf(out,"%u",&verbosity);
  if ((out=check_option(str,n,'o','o')) != NULL)
    if (strlen(out) > 0)
      outfile=out;
}

void print_progress(double peps, void *user_data)
{
  if (verbosity&VER_USR1)
    fprintf(stderr,"epsilon= %e\n",peps);
}

int main(int argc,char **argv)
{
  char stdi=0;
  unsigned int i,j,l;
  FILE *fout;
  LyapK *result;
  LyapKError error;

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
      check_alloc(outfile=(char*)calloc(strlen(infile)+6,1));
      sprintf(outfile,"%s.lyap",infile);
    }
    else {
      check_alloc(outfile=(char*)calloc(11,1));
      sprintf(outfile,"stdin.lyap");
    }
  }
  test_outfile(outfile);

  series=get_series(infile,&length,exclude,column,verbosity);

  result=lyap_k_compute(series,length,mindim,maxdim,delay,epsmin,epsmax,
			 eps0set,eps1set,epscount,reference,maxiter,window,
			 print_progress,NULL,&error);
  if (result == NULL) {
    if (error == LYAP_K_ERR_ZERO_INTERVAL) {
      /* interval == 0.0 only happens when every point of series equals
	 series[0], so series[0] stands in for both the original
	 rescale_data()'s *min and *min+*interval (which are then also
	 equal). */
      fprintf(stderr,"rescale_data: data ranges from %e to %e. It makes\n"
	      "\t\tno sense to continue. Exiting!\n\n",series[0],series[0]);
      exit(RESCALE_DATA_ZERO_INTERVAL);
    }
    else {
      fprintf(stderr,"Too few points to handle these parameters!\n");
      exit(LYAP_K__MAXITER_TOO_LARGE);
    }
  }

  fout=fopen(outfile,"w");
  if (verbosity&VER_INPUT)
    fprintf(stderr,"Opened %s for writing\n",outfile);
  for (l=0;l<result->epscount;l++) {
    for (i=0;i<result->maxdim-result->mindim+1;i++) {
      fprintf(fout,"#epsilon= %e  dim= %d\n",result->epsilon[l],
	      (int)(result->mindim+i));
      for (j=0;j<=result->maxiter;j++)
	if (result->count[l][i][j])
	  fprintf(fout,"%d %e %ld\n",j,
		  result->lyap[l][i][j]/result->count[l][i][j],
		  result->count[l][i][j]);
      fprintf(fout,"\n");
    }
    fflush(fout);
  }
  fclose(fout);

  lyap_k_free(result);

  return 0;
}
