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
/*Author: Stefan Caldararu*/
/*Changes:
  Jun 1, 2026: Convert ar-run.f to C
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>
#include <math.h>
#include "routines/tsa.h"
#include "../include/ar-run.h"

#define WID_STR "iterate AR model, e.g. as fitted by ar-model (Dresden)"

#define NP_MAX 100

unsigned long length=ULONG_MAX,exclude=0,ntrans=10000;
char *outfile=NULL,stdo=1;
char *infile=NULL;
unsigned int poles=NP_MAX;
unsigned int verbosity=1;
unsigned long seed=1L;

void show_options(char *progname)
{
  what_i_do(progname,WID_STR);
  fprintf(stderr," Usage: %s [options]\n",progname);
  fprintf(stderr," Options:\n");
  fprintf(stderr,"Everything not being a valid option will be interpreted"
          " as a possible datafile.\nIf no datafile is given stdin is read. "
          "Just - also means stdin\n");
  fprintf(stderr,"\t-l number of iterations (l=0: infinite)\n");
  fprintf(stderr,"\t-p order of AR-model (default determined from input)\n");
  fprintf(stderr,"\t-I seed for random numbers (If seed=0, a time-based seed"
          " is used) [Default: fixed]\n");
  fprintf(stderr,"\t-x number of transients discarded [default 10000]\n");
  fprintf(stderr,"\t-o output file name, just -o means ar.dat\n");
  fprintf(stderr,"\t-V verbosity level [default 1]\n\t\t"
          "0='only panic messages'\n\t\t"
          "1='+ input/output messages'\n");
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
  if ((out=check_option(argv,argc,'I','u')) != NULL) {
    sscanf(out,"%lu",&seed);
    if (seed == 0)
      seed=(unsigned long)time((time_t*)&seed);
  }
  if ((out=check_option(argv,argc,'x','u')) != NULL)
    sscanf(out,"%lu",&ntrans);
  if ((out=check_option(argv,argc,'V','u')) != NULL)
    sscanf(out,"%u",&verbosity);
  if ((out=check_option(argv,argc,'o','o')) != NULL) {
    stdo=0;
    if (strlen(out) > 0)
      outfile=out;
  }
}

int main(int argc,char **argv)
{
  char stdi=0,line[1024];
  FILE *file,*outf;
  long i,j,n,nn,np_read;
  double *x,*a,var,xx;
  
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
  
  /* Set up output file */
  if (outfile == NULL) {
    if (!stdi) {
      check_alloc(outfile=(char*)calloc(strlen(infile)+4,(size_t)1));
      strcpy(outfile,infile);
      strcat(outfile,".ar");
    }
    else {
      check_alloc(outfile=(char*)calloc((size_t)10,(size_t)1));
      strcpy(outfile,"ar.dat");
    }
  }
  if (!stdo)
    test_outfile(outfile);
  
  /* Allocate memory for AR model and state */
  check_alloc(a=(double*)malloc(sizeof(double)*NP_MAX));
  check_alloc(x=(double*)malloc(sizeof(double)*(NP_MAX+1)));
  
  /* Initialize state buffer */
  for (i=0;i<=NP_MAX;i++)
    x[i]=0.0;
  
  /* Open input file with AR coefficients */
  if (stdi) {
    file=stdin;
    if (verbosity&VER_INPUT)
      fprintf(stderr,"Reading from stdin\n");
  }
  else {
    file=fopen(infile,"r");
    if (file == NULL) {
      fprintf(stderr,"Can't open file %s\n",infile);
      exit(27);
    }
    if (verbosity&VER_INPUT)
      fprintf(stderr,"Reading AR coefficients from %s\n",infile);
  }
  
  /* Read first line to determine variance */
  if (fgets(line,1024,file) == NULL) {
    fprintf(stderr,"Error reading input file\n");
    exit(28);
  }
  
  if (line[0]=='#') {
    /* File has header with variance starting at column 18 */
    if (sscanf(line+17,"%lf",&var) != 1) {
      fprintf(stderr,"Cannot read variance from header\n");
      exit(29);
    }
    /* Now read coefficients */
    for (j=0;j<NP_MAX;j++) {
      if (fgets(line,1024,file) == NULL)
        break;
      if (line[0]=='#')
        break;
      if (sscanf(line,"%lf",&a[j]) != 1) {
        fprintf(stderr,"Error reading coefficient %ld\n",j+1);
        exit(30);
      }
    }
    np_read=j;
  }
  else {
    /* File has plain text format */
    if (sscanf(line,"%lf",&var) != 1) {
      fprintf(stderr,"Cannot read variance\n");
      exit(29);
    }
    /* Now read coefficients */
    for (j=0;j<NP_MAX;j++) {
      if (fgets(line,1024,file) == NULL)
        break;
      if (sscanf(line,"%lf",&a[j]) != 1)
        break;
    }
    np_read=j;
  }
  
  /* If poles not specified, use the number read from file */
  if (poles == NP_MAX)
    poles=np_read;
  
  if (poles > NP_MAX) {
    fprintf(stderr,"ar-run: order too large, increase NP_MAX\n");
    exit(31);
  }
  
  if (verbosity&VER_INPUT) {
    fprintf(stderr,"AR model order: %u\n",poles);
    fprintf(stderr,"Variance (driving amplitude): %e\n",var);
  }
  
  if (!stdi)
    fclose(file);
  
  /* Initialize random number generator */
  rnd_init(seed);
  
  /* Open output file */
  if (stdo)
    outf=stdout;
  else {
    outf=fopen(outfile,"w");
    if (outf == NULL) {
      fprintf(stderr,"Cannot open output file %s\n",outfile);
      exit(32);
    }
  }
  
  if (verbosity&VER_INPUT && !stdo)
    fprintf(stderr,"Writing output to %s\n",outfile);
  
  
  /* Iterate the AR model. length!=ULONG_MAX (an explicit -l) is handled by
     the reentrant ar_run_generate() API (source_c/api/ar_run_api.c), shared
     with the Python bindings. length==ULONG_MAX (no -l given) streams
     forever and can't be expressed as a bounded return, so it keeps its own
     loop below. */
  if (length != ULONG_MAX) {
    double *series=ar_run_generate(poles,a,var,length,ntrans,seed);
    for (n=0;n<(long)length;n++)
      fprintf(outf,"%e\n",series[n]);
    ar_run_free(series);
  }
  else {
    for (n=-(long)ntrans;;n++) {
      /* Compute next value using AR recurrence */
      nn=((n+ntrans)%poles);
      xx=gaussian(var);

      for (j=0;j<poles;j++) {
        int idx = (nn - j - 1 + poles) % poles;
        xx += a[j] * x[idx];
      }

      x[nn]=xx;

      /* Write output after transient phase */
      if (n>=0)
        fprintf(outf,"%e\n",xx);
    }
  }
  
  /* Close files and clean up */
  if (!stdo && outf != NULL)
    fclose(outf);
  
  free(a);
  free(x);
  
  return 0;
}