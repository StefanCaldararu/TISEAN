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
/*Author: Rainer Hegger Last modified: Nov 30, 2000 */
/*Changes:
  12/11/05: Going multivariate
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include "routines/tsa.h"
#include "../include/nrlazy.h"

#define WID_STR "Performs simple noise reduction."

unsigned long length=ULONG_MAX,exclude=0;
unsigned int comp=1,embed=5,delay=1,iterations=1;
unsigned int verbosity=0x3;
char *column=NULL;
double eps=1.0e-3,epsvar;

char *outfile=NULL,epsset=0,stdo=1,epsvarset=0;
char *infile=NULL;
double **series;
char dimset=0;

void show_options(char *progname)
{
  what_i_do(progname,WID_STR);
  fprintf(stderr," Usage: %s [Options]\n",progname);
  fprintf(stderr," Options:\n");
  fprintf(stderr,"Everything not being a valid option will be interpreted"
          " as a possible"
          " datafile.\nIf no datafile is given stdin is read. Just - also"
          " means stdin\n");
  fprintf(stderr,"\t-l # of data to use [default: whole file]\n");
  fprintf(stderr,"\t-x # of lines to be ignored [default: 0]\n");
  fprintf(stderr,"\t-c column to read [default: 1]\n");
  fprintf(stderr,"\t-m no. of comp.,embedding dim. [default: %u,%u]\n",
	  comp,embed);
  fprintf(stderr,"\t-d delay [default: 1]\n");
  fprintf(stderr,"\t-i iterations [default: 1]\n");
  fprintf(stderr,"\t-r neighborhoud size [default: (interval of data)/1000]\n");
  fprintf(stderr,"\t-v neighborhoud size (in units of the std. dev. of the "
	  "data \n\t\t(overwrites -r) [default: not set]\n");
  fprintf(stderr,"\t-o output file name [Default: 'datafile'.laz.n,"
	  "\n\t\twhere n is the number of the last iteration,"
	  "\n\t\twithout -o the last iteration is written to stdout.]\n");
  fprintf(stderr,"\t-V verbosity level [Default: 3]\n\t\t"
          "0='only panic messages'\n\t\t"
          "1='+ input/output messages'\n\t\t"
	  "2='+ write output of all iterations to files'\n\t\t"
	  "4='+ write the number of neighbors found for each point\n");
  fprintf(stderr,"\t-h show these options\n");
  exit(0);
}

void scan_options(int n,char **in)
{
  char *out;

  if ((out=check_option(in,n,'l','u')) != NULL)
    sscanf(out,"%lu",&length);
  if ((out=check_option(in,n,'x','u')) != NULL)
    sscanf(out,"%lu",&exclude);
  if ((out=check_option(in,n,'c','c')) != NULL) {
    column=out;
    dimset=1;
  }
  if ((out=check_option(in,n,'m','2')) != NULL)
    sscanf(out,"%u,%u",&comp,&embed);
  if ((out=check_option(in,n,'d','u')) != NULL)
    sscanf(out,"%u",&delay);
  if ((out=check_option(in,n,'i','u')) != NULL)
    sscanf(out,"%u",&iterations);
  if ((out=check_option(in,n,'r','f')) != NULL) {
    epsset=1;
    sscanf(out,"%lf",&eps);
  }
  if ((out=check_option(in,n,'V','u')) != NULL)
    sscanf(out,"%u",&verbosity);
  if ((out=check_option(in,n,'v','f')) != NULL) {
    epsvarset=1;
    sscanf(out,"%lf",&epsvar);
  }
  if ((out=check_option(in,n,'o','o')) != NULL) {
    stdo=0;
    if (strlen(out) > 0)
      outfile=out;
  }
}

/* Writes out one iteration's corrected data, exactly as the original
   inline loop inside main() used to: intermediate iterations only go to
   'outfile.iter' and only if VER_USR1 is set, while the last iteration
   also goes to stdout (if stdo) and/or 'outfile.iterations' (if !stdo or
   VER_USR1) - see nrlazy_correct()'s on_iteration callback. series is
   already scaled back to the original units, like nrlazy_correct()'s
   final result. user_data is the CLI's `ofname` scratch buffer. */
static void write_iteration(unsigned int iter,unsigned int iterations,
			     double *const *series,
			     const unsigned int *neighbors,void *user_data)
{
  char *ofname=(char*)user_data;
  unsigned long n;
  unsigned int i;
  FILE *file=NULL;

  if ((verbosity&VER_USR1) && (iter < iterations)) {
    sprintf(ofname,"%s.%d",outfile,iter);
    test_outfile(ofname);
    file=fopen(ofname,"w");
    if (verbosity&VER_INPUT)
      fprintf(stderr,"Opened %s for writing\n",ofname);
    if (stdo && (iter == iterations)) {
      if (verbosity&VER_INPUT)
	fprintf(stderr,"Writing to stdout\n");
    }
    for (n=0;n<length;n++) {
      if (stdo && (iter == iterations)) {
	if (verbosity&VER_USR2) {
	  for (i=0;i<comp;i++)
	    fprintf(stdout,"%e ",series[i][n]);
	  fprintf(stdout,"%u\n",neighbors[n]);
	}
	else {
	  fprintf(stdout,"%e",series[0][n]);
	  for (i=1;i<comp;i++)
	    fprintf(stdout,"%e ",series[i][n]);
	  fprintf(stdout,"\n");
	}
      }
      if (verbosity&VER_USR2) {
	for (i=0;i<comp;i++)
	  fprintf(file,"%e ",series[i][n]);
	fprintf(file,"%u\n",neighbors[n]);
      }
      else {
	fprintf(file,"%e",series[0][n]);
	for (i=1;i<comp;i++)
	  fprintf(file," %e",series[i][n]);
	fprintf(file,"\n");
      }
    }
    fclose(file);
  }
  if (iter == iterations) {
    if (!stdo || (verbosity&VER_USR1)) {
      sprintf(ofname,"%s.%d",outfile,iter);
      test_outfile(ofname);
      file=fopen(ofname,"w");
      if (verbosity&VER_INPUT)
	fprintf(stderr,"Opened %s for writing\n",ofname);
      if (stdo && (iter == iterations)) {
	if (verbosity&VER_INPUT)
	  fprintf(stderr,"Writing to stdout\n");
      }
    }
    for (n=0;n<length;n++) {
      if (stdo) {
	if (verbosity&VER_USR2) {
	  for (i=0;i<comp;i++)
	    fprintf(stdout,"%e ",series[i][n]);
	  fprintf(stdout,"%u\n",neighbors[n]);
	}
	else {
	  fprintf(stdout,"%e",series[0][n]);
	  for (i=1;i<comp;i++)
	    fprintf(stdout," %e",series[i][n]);
	  fprintf(stdout,"\n");
	}
      }
      if (!stdo || (verbosity&VER_USR1)) {
	if (verbosity&VER_USR2) {
	  for (i=0;i<comp;i++)
	    fprintf(file,"%e ",series[i][n]);
	  fprintf(file,"%u\n",neighbors[n]);
	}
	else {
	  fprintf(file,"%e",series[0][n]);
	  for (i=1;i<comp;i++)
	    fprintf(file," %e",series[i][n]);
	  fprintf(file,"\n");
	}
      }
    }
    if (!stdo || (verbosity&VER_USR1))
      fclose(file);
  }
}

int main(int argc,char **argv)
{
  char *ofname;
  char stdi=0;
  unsigned long i;
  double bad_value=0.0;
  NRLazyResult *result;

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
      check_alloc(ofname=(char*)calloc(strlen(infile)+9,(size_t)1));
      sprintf(outfile,"%s.laz",infile);
    }
    else {
      check_alloc(outfile=(char*)calloc((size_t)10,(size_t)1));
      check_alloc(ofname=(char*)calloc((size_t)14,(size_t)1));
      sprintf(outfile,"stdin.laz");
    }
  }
  else
    check_alloc(ofname=(char*)calloc(strlen(outfile)+10,(size_t)1));


  if (column == NULL)
    series=(double**)get_multi_series(infile,&length,exclude,&comp,"",dimset,
				      verbosity);
  else
    series=(double**)get_multi_series(infile,&length,exclude,&comp,column,
				      dimset,verbosity);

  result=nrlazy_correct((double *const *)series,length,comp,embed,delay,
			 iterations,epsset?eps:(double)NAN,
			 epsvarset?epsvar:(double)NAN,&bad_value,
			 write_iteration,ofname);
  if (result == NULL) {
    fprintf(stderr,"rescale_data: data ranges from %e to %e. It makes\n"
	    "\t\tno sense to continue. Exiting!\n\n",bad_value,bad_value);
    exit(RESCALE_DATA_ZERO_INTERVAL);
  }
  nrlazy_free(result);

  /*cleaning up */
  for (i=0;i<comp;i++)
    free(series[i]);
  free(series);

  if (outfile != NULL)
    free(outfile);
  if (ofname != NULL)
    free(ofname);
  if (infile != NULL)
    free(infile);
  if (column != NULL)
    free(column);
  /* end cleaning up */

  return 0;
}
