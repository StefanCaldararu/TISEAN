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
/*Author: Rainer Hegger. Last modified: Dec 10, 2005 */
/*Changes:
  12/10/05: It's multivariate now
  12/16/05: Scaled <eps> and sigma(eps)
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include "routines/tsa.h"
#include "../include/false_nearest.h"

#define WID_STR "Determines the fraction of false nearest neighbors."

char *outfile=NULL;
char *infile=NULL;
char stdo=1,dimset=0;
char *column=NULL;
unsigned long length=ULONG_MAX,exclude=0,theiler=0;
unsigned int delay=1,minemb=1;
unsigned int comp=1,maxemb=5;
unsigned int verbosity=0xff;
double rt=2.0;
double eps0=1.0e-5;
double **series;

void show_options(char *progname)
{
  what_i_do(progname,WID_STR);
  fprintf(stderr," Usage: %s [options]\n",progname);
  fprintf(stderr," Options:\n");
  fprintf(stderr,"Everything not being a valid option will be interpreted"
          " as a possible"
          " datafile.\nIf no datafile is given stdin is read. Just - also"
          " means stdin\n");
  fprintf(stderr,"\t-l # of data [default: whole file]\n");
  fprintf(stderr,"\t-x # of lines to ignore [default: 0]\n");
  fprintf(stderr,"\t-c columns to read [default: 1]\n");
  fprintf(stderr,"\t-m min. test embedding dimension [default: %u]\n",minemb);
  fprintf(stderr,"\t-M # of components,max. emb. dim. [default: %u,%u]\n",
	  comp,maxemb);
  fprintf(stderr,"\t-d delay [default: 1]\n");
  fprintf(stderr,"\t-f escape factor [default: %.2lf]\n",rt);
  fprintf(stderr,"\t-t theiler window [default: 0]\n");
  fprintf(stderr,"\t-o output file [default: 'datafile'.fnn; without -o"
	  " stdout]\n");
  fprintf(stderr,"\t-V verbosity level [default: 3]\n\t\t"
          "0='only panic messages'\n\t\t"
          "1='+ input/output messages'\n\t\t"
          "2='+ information about the current state\n");
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
  if ((out=check_option(in,n,'c','s')) != NULL)
    column=out;
  if ((out=check_option(in,n,'m','u')) != NULL)
    sscanf(out,"%u",&minemb);
  if ((out=check_option(in,n,'M','2')) != NULL) {
    sscanf(out,"%u,%u",&comp,&maxemb);
    dimset=1;
  }
  if ((out=check_option(in,n,'d','u')) != NULL)
    sscanf(out,"%u",&delay);
  if ((out=check_option(in,n,'f','f')) != NULL)
    sscanf(out,"%lf",&rt);
  if ((out=check_option(in,n,'t','u')) != NULL)
    sscanf(out,"%lu",&theiler);
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
  FILE *file=NULL;
  unsigned long i,k;
  unsigned int c;
  double cmin,cinterval;
  FalseNearest *result;
  FalseNearestError error;

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
      strcat(outfile,".fnn");
    }
    else {
      check_alloc(outfile=(char*)calloc((size_t)10,(size_t)1));
      strcpy(outfile,"stdin.fnn");
    }
  }
  if (!stdo)
    test_outfile(outfile);

  if (column == NULL)
    series=(double**)get_multi_series(infile,&length,exclude,&comp,"",dimset,
				      verbosity);
  else
    series=(double**)get_multi_series(infile,&length,exclude,&comp,column,
				      dimset,verbosity);

  if (!stdo) {
    file=fopen(outfile,"w");
    if (verbosity&VER_INPUT)
      fprintf(stderr,"Opened %s for writing\n",outfile);
  }
  else {
    if (verbosity&VER_INPUT)
      fprintf(stderr,"Writing to stdout\n");
  }

  result=false_nearest_compute((double *const *)series,length,comp,delay,
				minemb,maxemb,theiler,rt,eps0,&error);
  if (result == NULL) {
    if (error == FALSE_NEAREST_ERR_ZERO_INTERVAL) {
      /* Reproduce rescale_data()'s exact message for whichever component
	 first has zero range. false_nearest_compute() rescales a private
	 copy and doesn't report which component failed, but it never
	 touches our own copy of series, so redo the same scan here. */
      for (c=0;c<comp;c++) {
	cmin=cinterval=series[c][0];
	for (k=1;k<length;k++) {
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
    else if (error == FALSE_NEAREST_ERR_ZERO_VARIANCE) {
      fprintf(stderr,"Variance of the data is zero. Exiting!\n\n");
      exit(VARIANCE_VAR_EQ_ZERO);
    }
    else {
      fprintf(stderr,"Not enough points found!\n");
      exit(FALSE_NEAREST_NOT_ENOUGH_POINTS);
    }
  }

  for (i=0;i<result->n;i++) {
    if (stdo) {
      fprintf(stdout,"%u %e %e %e\n",result->dimension[i],result->fraction[i],
	      result->avg_eps[i],result->sigma_eps[i]);
      fflush(stdout);
    }
    else {
      fprintf(file,"%u %e %e %e\n",result->dimension[i],result->fraction[i],
	      result->avg_eps[i],result->sigma_eps[i]);
      fflush(file);
    }
  }
  if (!stdo)
    fclose(file);

  false_nearest_free(result);
  if (infile != NULL)
    free(infile);
  if (outfile != NULL)
    free(outfile);
  free(series);

  return 0;
}
