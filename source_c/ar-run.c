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
/*Author: T. Schreiber (1999). C port.*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "routines/tsa.h"

#define WID_STR "Iterates an AR model, e.g. as fitted by ar-model"
#define NPMAX 100

char *infile=NULL,*outfile=NULL,onscreen=1;
unsigned int poles=0,polesset=0;
unsigned int verbosity=1;
long ntrans=10000;
unsigned long nmax=0,nmaxset=0,seed=0x44325;

void show_options(char *progname)
{
  what_i_do(progname,WID_STR);
  fprintf(stderr," Usage: %s -l# [-p# -I# -o outfile -x# -V# -h] file\n",
	  progname);
  fprintf(stderr," Options:\n");
  fprintf(stderr,"\t-l # of iterations (l=0: infinite) [required]\n");
  fprintf(stderr,"\t-p order of the AR-model [default determined from"
	  " input]\n");
  fprintf(stderr,"\t-I seed for the random numbers\n");
  fprintf(stderr,"\t-x # of transients discarded [default 10000]\n");
  fprintf(stderr,"\t-o output file name, just -o means ar.dat\n");
  fprintf(stderr,"\t-V verbosity level [default 1]\n");
  fprintf(stderr,"\t-h show this message\n\n");
  fprintf(stderr,"The coefficients are read from file. The format is one"
	  " line containing\nthe rms amplitude of the increments, followed"
	  " by one line for each of\nthe coefficients.\n\n");
  exit(0);
}

void scan_options(int argc,char **argv)
{
  char *out;

  if ((out=check_option(argv,argc,'p','u')) != NULL) {
    sscanf(out,"%u",&poles);
    polesset=1;
  }
  if ((out=check_option(argv,argc,'l','u')) != NULL) {
    sscanf(out,"%lu",&nmax);
    nmaxset=1;
  }
  if ((out=check_option(argv,argc,'x','d')) != NULL)
    sscanf(out,"%ld",&ntrans);
  if ((out=check_option(argv,argc,'I','u')) != NULL)
    sscanf(out,"%lu",&seed);
  if ((out=check_option(argv,argc,'V','u')) != NULL)
    sscanf(out,"%u",&verbosity);
  if ((out=check_option(argv,argc,'o','o')) != NULL) {
    onscreen=0;
    if (strlen(out) > 0)
      outfile=out;
  }
}

double ar_step(double *x,unsigned int poles,double *a,double var,long t)
{
  long idx=((t%(long)poles)+(long)poles)%(long)poles;
  double xx=gaussian(var);
  unsigned int j;

  for (j=1;j<=poles;j++) {
    long jidx=(((t-(long)j)%(long)poles)+(long)poles)%(long)poles;
    xx += a[j-1]*x[jidx];
  }
  x[idx]=xx;

  return xx;
}

int main(int argc,char **argv)
{
  FILE *fin,*fout;
  char *line;
  int input_size;
  unsigned int i,j,requested;
  double var,a[NPMAX],*x;
  long n;

  if (scan_help(argc,argv))
    show_options(argv[0]);

  scan_options(argc,argv);
  if (!nmaxset) {
    fprintf(stderr,"ar-run: the -l option is mandatory!\n");
    exit(1);
  }
  if (polesset && (poles > NPMAX)) {
    fprintf(stderr,"ar-run: make NPMAX larger.\n");
    exit(1);
  }
#ifndef OMIT_WHAT_I_DO
  if (verbosity&VER_INPUT)
    what_i_do(argv[0],WID_STR);
#endif

  infile=search_datafile(argc,argv,NULL,verbosity);
  if (infile == NULL)
    fin=stdin;
  else {
    fin=fopen(infile,"r");
    if (fin == NULL) {
      fprintf(stderr,"ar-run: could not open %s for reading!\n",infile);
      exit(1);
    }
  }

  if (!onscreen && (outfile == NULL))
    outfile="ar.dat";
  if (!onscreen)
    test_outfile(outfile);

  requested=polesset?poles:NPMAX;

  input_size=INPUT_SIZE;
  check_alloc(line=(char*)calloc((size_t)input_size,(size_t)1));

  if (myfgets(line,&input_size,fin,verbosity) == NULL) {
    fprintf(stderr,"wrong input format! try:\n(rms of increments)\na(1)"
	    "\na(2)\n...\n");
    exit(1);
  }
  if (sscanf(line,"%lf",&var) != 1) {
    fprintf(stderr,"wrong input format! try:\n(rms of increments)\na(1)"
	    "\na(2)\n...\n");
    exit(1);
  }

  poles=0;
  for (j=0;j<requested;j++) {
    if (myfgets(line,&input_size,fin,verbosity) == NULL)
      break;
    if (sscanf(line,"%lf",&a[j]) != 1)
      break;
    poles++;
  }
  if (fin != stdin)
    fclose(fin);
  free(line);

  if (poles == 0) {
    fprintf(stderr,"wrong input format! try:\n(rms of increments)\na(1)"
	    "\na(2)\n...\n");
    exit(1);
  }

  if (verbosity&VER_INPUT) {
    fprintf(stderr,"coefficients:      ");
    for (i=0;i<poles;i++)
      fprintf(stderr,"%e ",a[i]);
    fprintf(stderr,"\ndriving amplitude: %e\n",var);
  }

  if (onscreen)
    fout=stdout;
  else {
    fout=fopen(outfile,"w");
    if (verbosity&VER_INPUT)
      fprintf(stderr,"Opened %s for writing\n",outfile);
  }

  check_alloc(x=(double*)calloc((size_t)poles,(size_t)sizeof(double)));

  rnd_init(seed);

  for (n=0;n<ntrans;n++)
    ar_step(x,poles,a,var,n);

  if (nmax == 0) {
    for (n=ntrans;;n++)
      fprintf(fout,"%e\n",ar_step(x,poles,a,var,n));
  }
  else {
    for (n=ntrans;n<ntrans+(long)nmax;n++)
      fprintf(fout,"%e\n",ar_step(x,poles,a,var,n));
  }

  if (!onscreen)
    fclose(fout);

  free(x);
  if (infile != NULL)
    free(infile);

  return 0;
}
