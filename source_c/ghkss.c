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
/*Author: Rainer Hegger Last modified: Jun 10, 2006 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include "routines/tsa.h"
#include "../include/ghkss.h"

#define WID_STR "Multivariate noise reduction using the GHKSS algorithm"


unsigned long length=ULONG_MAX,exclude=0;
unsigned int qdim=2,delay=1,minn=50,iterations=1,comp=1,embed=5;
unsigned int verbosity=0xff;
double mineps;
char *column=NULL;
char eps_set=0,euclidean=0,dimset=0;
char *outfile=NULL,stdo=1;
char *infile=NULL;

double **series;

void show_options(char *progname)
{
  what_i_do(progname,WID_STR);
  fprintf(stderr,"Usage: %s [options]\n",progname);
  fprintf(stderr,"Options:\n");
  fprintf(stderr,"Everything not being a valid option will be interpreted"
          " as a possible"
          " datafile.\nIf no datafile is given stdin is read. Just - also"
          " means stdin\n");
  fprintf(stderr,"\t-l # of data to use [Default: whole file]\n");
  fprintf(stderr,"\t-x # of lines to be ignored [Default: 0]\n");
  fprintf(stderr,"\t-c column to read [Default: 1,..,# of components]\n");
  fprintf(stderr,"\t-m # of components,embedding dimension [Default: 1,5]\n");
  fprintf(stderr,"\t-d delay [Default: 1]\n");
  fprintf(stderr,"\t-q dimension to project to [Default: 2]\n");
  fprintf(stderr,"\t-k minimal number of neighbours [Default: 50]\n");
  fprintf(stderr,"\t-r minimal neighbourhood size \n\t\t"
	  "[Default: (interval of data)/1000]\n");
  fprintf(stderr,"\t-i # of iterations [Default: 1]\n");
  fprintf(stderr,"\t-2 use euklidean metric [Default: non euklidean]\n");
  fprintf(stderr,"\t-o name of output file \n\t\t"
	  "[Default: 'datafile'.opt.n, where n is the iteration.\n\t\t"
	  " If no -o is given, the last iteration is also"
	  " written to stdout]\n");
  fprintf(stderr,"\t-V verbosity level [Default: 7]\n\t\t"
          "0='only panic messages'\n\t\t"
          "1='+ input/output messages'\n\t\t"
          "2='+ average correction and trend'\n\t\t"
	  "4='+ how many points for which epsilon'\n");
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
  if ((out=check_option(in,n,'c','s')) != NULL) {
    column=out;
    dimset=1;
  }
  if ((out=check_option(in,n,'m','2')) != NULL)
    sscanf(out,"%u,%u",&comp,&embed);
  if ((out=check_option(in,n,'d','u')) != NULL)
    sscanf(out,"%u",&delay);
  if ((out=check_option(in,n,'q','u')) != NULL)
    sscanf(out,"%u",&qdim);
  if ((out=check_option(in,n,'k','u')) != NULL)
    sscanf(out,"%u",&minn);
  if ((out=check_option(in,n,'r','f')) != NULL) {
    eps_set=1;
    sscanf(out,"%lf",&mineps);
  }
  if ((out=check_option(in,n,'i','u')) != NULL)
    sscanf(out,"%u",&iterations);
  if ((out=check_option(in,n,'V','u')) != NULL)
    sscanf(out,"%u",&verbosity);
  if ((out=check_option(in,n,'2','n')) != NULL)
    euclidean=1;
  if ((out=check_option(in,n,'o','o')) != NULL) {
    stdo=0;
    if (strlen(out) > 0)
      outfile=out;
  }
}

int main(int argc,char **argv)
{
  char stdi=0;
  int iter;
  long i,j;
  unsigned long s;
  char *ofname;
  FILE *file;
  GHKSSResult *result;
  GHKSSIteration *it;
  GHKSSError error;
  double bad_value;

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
      sprintf(outfile,"%s.opt",infile);
    }
    else {
      check_alloc(outfile=(char*)calloc((size_t)10,(size_t)1));
      check_alloc(ofname=(char*)calloc((size_t)14,(size_t)1));
      sprintf(outfile,"stdin.opt");
    }
  }
  else
    check_alloc(ofname=(char*)calloc(strlen(outfile)+10,(size_t)1));

  if (column == NULL)
    series=(double**)get_multi_series(infile,&length,exclude,&comp,"",
				     dimset,verbosity);
  else
    series=(double**)get_multi_series(infile,&length,exclude,&comp,column,
				      dimset,verbosity);

  result=ghkss_reduce((double *const *)series,length,comp,embed,delay,qdim,
		       minn,mineps,eps_set,iterations,euclidean,
		       &error,&bad_value);
  if (result == NULL) {
    switch (error) {
    case GHKSS_ERR_TOO_MANY_NEIGHBORS:
      fprintf(stderr,"With %lu data you will never find %u neighbors."
	      " Exiting!\n",length,minn);
      exit(GHKSS__TOO_MANY_NEIGHBORS);
    case GHKSS_ERR_ZERO_INTERVAL:
      fprintf(stderr,"rescale_data: data ranges from %e to %e. It makes\n"
	      "\t\tno sense to continue. Exiting!\n\n",bad_value,bad_value);
      exit(RESCALE_DATA_ZERO_INTERVAL);
    case GHKSS_ERR_EIGEN_NO_CONVERGE:
    default:
      fprintf(stderr,"Non converging eigenvalues! Exiting\n");
      exit(EIG2_TOO_MANY_ITERATIONS);
    }
  }

  for (iter=1;iter<=iterations;iter++) {
    it=&result->iters[iter-1];

    if (verbosity&(VER_USR1|VER_USR2))
      fprintf(stderr,"Starting iteration %d\n",iter);

    if (verbosity&VER_USR2)
      for (s=0;s<it->n_correction_steps;s++)
	fprintf(stderr,"Corrected %ld points with epsilon= %e\n",
		it->correction_steps[s].count,it->correction_steps[s].epsilon);

    if (verbosity&VER_USR2)
      fprintf(stderr,"Start evaluating the trend\n");

    if (verbosity&VER_USR2)
      for (s=0;s<it->n_trend_steps;s++)
	fprintf(stderr,"Trend subtracted for %ld points with epsilon= %e\n",
		it->trend_steps[s].count,it->trend_steps[s].epsilon);

    if (verbosity&(VER_USR1|VER_USR2))
      for (i=0;i<comp;i++) {
	fprintf(stderr,"Average shift of component %ld = %e\n",i+1,
		it->shift[i]);
	fprintf(stderr,"Average rms correction of comp. %ld = %e\n\n",
		i+1,it->rms[i]);
      }

    if (it->mineps_reset && (verbosity&VER_USR2))
      fprintf(stderr,"Reset minimal neighbourhood size to %e\n",
	      it->mineps_after);

    sprintf(ofname,"%s.%d",outfile,iter);
    test_outfile(ofname);

    file=fopen(ofname,"w");
    if (verbosity&VER_INPUT)
      fprintf(stderr,"Opened %s for writing\n\n",ofname);
    for (i=0;i<length;i++) {
      for (j=0;j<comp;j++) {
	fprintf(file,"%e ",it->series[j][i]);
      }
      fprintf(file,"\n");
      if (stdo && (iter == iterations)) {
	for (j=0;j<comp;j++)
	  fprintf(stdout,"%e ",it->series[j][i]);
	fprintf(stdout,"\n");
      }
    }
    fclose(file);
  }

  ghkss_free(result);

  return 0;
}
