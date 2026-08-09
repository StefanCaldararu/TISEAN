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
/*Author: Rainer Hegger */
/*Changes:
  Sep 8, 2006: Add -o functionality
  Sep 7, 2006: Completely rewritten to handle multivariate data
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "routines/tsa.h"
#include "../include/lfo-test.h"
#include <math.h>

#define WID_STR "Estimates the average forecast error of a local\n\t\
linear fit"

double **series;

char epsset=0,causalset=0;
unsigned int verbosity=VER_INPUT|VER_FIRST_LINE;
unsigned int COMP=1,EMBED=2,DELAY=1,MINN=30,STEP=1;
double EPS0=1.e-3,EPSF=1.2;
unsigned long LENGTH=ULONG_MAX,exclude=0,CLENGTH=ULONG_MAX,causal;
char *infile=NULL,*COLUMN=NULL,*outfile=NULL;
char dimset=0,stout=1;

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
  fprintf(stderr,"\t-c columns to read [default: 1]\n");
  fprintf(stderr,"\t-m # of components, embedding dimension "
	  "[default: %u,%u]\n",COMP,EMBED);
  fprintf(stderr,"\t-d delay [default: 1]\n");
  fprintf(stderr,"\t-n iterations [default: length]\n");
  fprintf(stderr,"\t-k minimal number of neighbors for the fit "
	  "[default: 30]\n");
  fprintf(stderr,"\t-r neighborhoud size to start with "
	  "[default: (data interval)/1000]\n");
  fprintf(stderr,"\t-f factor to increase size [default: 1.2]\n");
  fprintf(stderr,"\t-s steps to forecast [default: 1]\n");
  fprintf(stderr,"\t-C width of causality window [default: steps]\n");
  fprintf(stderr,"\t-o output file [default 'datafile'.fce"
	  " no -o means write to stdout]\n");
  fprintf(stderr,"\t-V verbosity level [default: 1]\n\t\t"
          "0='only panic messages'\n\t\t"
          "1='+ input/output messages'\n\t\t"
	  "2='+ print indiviual forecast errors'\n");
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
  if ((out=check_option(in,n,'c','s')) != NULL) {
    COLUMN=out;
    dimset=1;
  }
  if ((out=check_option(in,n,'m','2')) != NULL)
    sscanf(out,"%u,%u",&COMP,&EMBED);
  if ((out=check_option(in,n,'d','u')) != NULL)
    sscanf(out,"%u",&DELAY);
  if ((out=check_option(in,n,'n','u')) != NULL)
    sscanf(out,"%lu",&CLENGTH);
  if ((out=check_option(in,n,'V','u')) != NULL)
    sscanf(out,"%u",&verbosity);
  if ((out=check_option(in,n,'k','u')) != NULL)
    sscanf(out,"%u",&MINN);
  if ((out=check_option(in,n,'r','f')) != NULL) {
    epsset=1;
    sscanf(out,"%lf",&EPS0);
  }
  if ((out=check_option(in,n,'f','f')) != NULL)
    sscanf(out,"%lf",&EPSF);
  if ((out=check_option(in,n,'s','u')) != NULL)
    sscanf(out,"%u",&STEP);
  if ((out=check_option(in,n,'C','u')) != NULL) {
    sscanf(out,"%lu",&causal);
    causalset=1;
  }
  if ((out=check_option(in,n,'o','o')) != NULL) {
    stout=0;
    if (strlen(out) > 0)
      outfile=out;
  }
}

int main(int argc,char **argv)
{
  char stin=0;
  long i,j;
  unsigned long clength;
  double bad_value;
  LfoTest *result;
  FILE *fout;

  if (scan_help(argc,argv))
    show_options(argv[0]);

  scan_options(argc,argv);

  if (!causalset)
    causal=STEP;

#ifndef OMIT_WHAT_I_DO
  if (verbosity&VER_INPUT)
    what_i_do(argv[0],WID_STR);
#endif

  infile=search_datafile(argc,argv,NULL,verbosity);
  if (infile == NULL)
    stin=1;

  if (outfile == NULL) {
    if (!stin) {
      check_alloc(outfile=(char*)calloc(strlen(infile)+5,(size_t)1));
      strcpy(outfile,infile);
      strcat(outfile,".fce");
    }
    else {
      check_alloc(outfile=(char*)calloc((size_t)10,(size_t)1));
      strcpy(outfile,"stdin.fce");
    }
  }
  if (!stout)
    test_outfile(outfile);

  if (COLUMN == NULL)
    series=(double**)get_multi_series(infile,&LENGTH,exclude,&COMP,"",dimset,
                                      verbosity);
  else
    series=(double**)get_multi_series(infile,&LENGTH,exclude,&COMP,COLUMN,
                                      dimset,verbosity);

  if ((LENGTH-(EMBED-1)*DELAY) < MINN) {
    fprintf(stderr,"Data set is too short to find enough neighbors "
	    "for the fit! Exiting!\n");
    exit(ONESTEP_TOO_FEW_POINTS);
  }

  result=lfo_test_forecast((double *const *)series,LENGTH,COMP,EMBED,DELAY,
			    MINN,STEP,causal,CLENGTH,EPS0,epsset,EPSF,
			    &bad_value);
  if (result == NULL) {
    fprintf(stderr,"rescale_data: data ranges from %e to %e. It makes\n"
	    "\t\tno sense to continue. Exiting!\n\n",bad_value,bad_value);
    exit(RESCALE_DATA_ZERO_INTERVAL);
  }

  clength=(CLENGTH <= LENGTH) ? CLENGTH-STEP : LENGTH-STEP;

  if (stout) {
    if (verbosity&VER_USR1) {
      fprintf(stdout,"#Relative forecast errors for each component:\n");
      for (i=0;i<COMP;i++)
	fprintf(stdout,"# %e\n",result->rms_error[i]);

      for (i=(EMBED-1)*DELAY;i<(long)clength;i++) {
	for (j=0;j<COMP-1;j++)
	  fprintf(stdout,"%e ",result->individual[j*LENGTH+i]);
	fprintf(stdout,"%e\n",result->individual[(COMP-1)*LENGTH+i]);
      }
    }
    else {
      fprintf(stdout,"#Relative forecast errors for each component:\n");
      for (i=0;i<COMP;i++)
	fprintf(stdout,"%e\n",result->rms_error[i]);
    }
  }
  else {
    fout=fopen(outfile,"w");
    if (verbosity&VER_INPUT)
      fprintf(stderr,"Opened %s for writing\n",outfile);
    if (verbosity&VER_USR1) {
      fprintf(fout,"#Relative forecast errors for each component:\n");
      for (i=0;i<COMP;i++)
	fprintf(fout,"# %e\n",result->rms_error[i]);

      for (i=(EMBED-1)*DELAY;i<(long)clength;i++) {
	for (j=0;j<COMP-1;j++)
	  fprintf(fout,"%e ",result->individual[j*LENGTH+i]);
	fprintf(fout,"%e\n",result->individual[(COMP-1)*LENGTH+i]);
      }
    }
    else {
      fprintf(fout,"#Relative forecast errors for each component:\n");
      for (i=0;i<COMP;i++)
	fprintf(fout,"%e\n",result->rms_error[i]);
    }
    fclose(fout);
    free(outfile);
  }

  lfo_test_free(result);

  return 0;
}
