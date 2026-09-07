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
  Feb 19, 2007: changed meaning of -f flag and added -P flag to be 
                consistent with spectrum
  Dec 5, 2006: Seg fault when poles > length;
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include "routines/tsa.h"
#include "../include/mem_spec.h"
#include <math.h>

#define WID_STR "Estimates the power spectrum of the data"

#ifndef M_PI
#define M_PI 3.1415926535897932385E0
#endif

unsigned long poles=128,out=2000;
unsigned long length=ULONG_MAX,exclude=0;
unsigned int column=1;
unsigned int verbosity=0x1;
double samplingrate=1.0;
char *outfile=NULL,stdo=1;
char *infile=NULL;
double *series;

void show_options(char *progname)
{
  what_i_do(progname,WID_STR);
  fprintf(stderr," Usage: %s [Options]\n",progname);
  fprintf(stderr," Options:\n");
  fprintf(stderr,"Everything not being a valid option will be interpreted"
          " as a possible"
          " datafile.\nIf no datafile is given stdin is read. Just - also"
          " means stdin\n");
  fprintf(stderr,"\t-l length of file [default is whole file]\n");
  fprintf(stderr,"\t-x # of lines to be ignored [default is 0]\n");
  fprintf(stderr,"\t-c column to read [default is 1]\n");
  fprintf(stderr,"\t-p number of poles [default is 128 or file length]\n");
  fprintf(stderr,"\t-P number of frequences out [default is 2000]\n");
  fprintf(stderr,"\t-f sampling rate in Hz [default is 1]\n");
  fprintf(stderr,"\t-o outfile [default is 'datafile'.spec]\n");
  fprintf(stderr,"\t-V verbosity level [default is 1]\n\t\t"
          "0='only panic messages'\n\t\t"
          "1='+ input/output messages'\n\t\t"
          "2='+ print the ar coefficients too'\n");
  fprintf(stderr,"\t-h show these options\n\n");
  exit(0);
}

void scan_options(int argc,char **argv)
{
  char *hout;
  
  if ((hout=check_option(argv,argc,'l','u')) != NULL)
    sscanf(hout,"%lu",&length);
  if ((hout=check_option(argv,argc,'x','u')) != NULL)
    sscanf(hout,"%lu",&exclude);
  if ((hout=check_option(argv,argc,'c','u')) != NULL)
    sscanf(hout,"%u",&column);
  if ((hout=check_option(argv,argc,'p','u')) != NULL)
    sscanf(hout,"%lu",&poles);
  if ((hout=check_option(argv,argc,'P','u')) != NULL)
    sscanf(hout,"%lu",&out);
  if ((hout=check_option(argv,argc,'f','f')) != NULL)
    sscanf(hout,"%lf",&samplingrate);
  if ((hout=check_option(argv,argc,'V','u')) != NULL)
    sscanf(hout,"%u",&verbosity);
  if ((hout=check_option(argv,argc,'o','o')) != NULL) {
    stdo=0;
    if (strlen(hout) > 0)
      outfile=hout;
  }
}

int main(int argc,char **argv)
{
  char stdi=0;
  double pm,*cof,*freq,*spec;
  long i;
  FILE *fout;
  MemSpecModel *model;

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
      check_alloc(outfile=(char*)calloc(strlen(infile)+6,(size_t)1));
      strcpy(outfile,infile);
      strcat(outfile,".spec");
    }
    else {
      check_alloc(outfile=(char*)calloc((size_t)11,(size_t)1));
      strcpy(outfile,"stdin.spec");
    }
  }
  if (!stdo)
    test_outfile(outfile);

  series=(double*)get_series(infile,&length,exclude,column,verbosity);

  if (length <= poles) {
    fprintf(stderr,"\n\tNo. of poles has to be smaller then the length of the\n"
	    "\tdata set! Exiting.\n");
    exit(MEM_SPEC_TOO_MANY_POLES);
  }

  model=mem_spec_fit(series,length,poles);
  if (model == NULL) {
    fprintf(stderr,"Variance of the data is zero. Exiting!\n\n");
    exit(VARIANCE_VAR_EQ_ZERO);
  }
  pm=model->sigma2;
  cof=model->coef;

  check_alloc(freq=(double*)malloc(sizeof(double)*out));
  check_alloc(spec=(double*)malloc(sizeof(double)*out));
  mem_spec_spectrum(model,out,samplingrate,freq,spec);

  if (!stdo) {
    fout=fopen(outfile,"w");
    if (verbosity&VER_INPUT)
      fprintf(stderr,"Opened %s for writing\n",outfile);
    if (verbosity&VER_USR1) {
      fprintf(fout,"#sigma^2=%e\n",pm);
      for (i=0;i<poles;i++)
	fprintf(fout,"#%ld %e\n",i+1,cof[i]);
    }
    for(i=0;i<out;i++) {
      fprintf(fout,"%e %e\n",freq[i],
	      spec[i]/sqrt((double)length));
      fflush(fout);
    }
    fclose(fout);
  }
  else {
    if (verbosity&VER_INPUT)
      fprintf(stderr,"Writing to stdout\n");
    if (verbosity&VER_USR1) {
      fprintf(stdout,"#sigma^2=%e\n",pm);
      for (i=0;i<poles;i++)
	fprintf(stdout,"#%ld %e\n",i+1,cof[i]);
    }
    for(i=0;i<out;i++) {
      fprintf(stdout,"%e %e\n",freq[i],
	      spec[i]/*/sqrt((double)length)*/);
    }
  }

  mem_spec_free(model);
  free(freq);
  free(spec);

  return 0;
}

