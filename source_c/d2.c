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
/*Author: Rainer Hegger. Last modified May 10, 2000 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include "routines/tsa.h"
#include "../include/d2.h"

#define WID_STR "Estimates the correlation sum, -dimension and -entropy"

/* output is written every WHEN seconds */
#define WHEN 120

char dimset=0,rescale_set=0,eps_min_set=0,eps_max_set=0;
char *FOUT=NULL;
double EPSMAX=1.0,EPSMIN=1.e-3;
unsigned long MINDIST=0,MAXFOUND=1000;
unsigned long length=ULONG_MAX,exclude=0;
unsigned int DIM=1,EMBED=10,HOWOFTEN=100,DELAY=1;
unsigned int verbosity=0x1;
char *column=NULL;
char *infile=NULL;

void show_options(char *progname)
{
  what_i_do(progname,WID_STR);
  fprintf(stderr,"  Usage: %s [options]\n",progname);
  fprintf(stderr,"  Options:\n");
  fprintf(stderr,"Everything not being a valid option will be interpreted"
          " as a possible"
          " datafile.\nIf no datafile is given stdin is read. Just - also"
          " means stdin\n");
  fprintf(stderr,"\t-l datapoints [default is whole file]\n");
  fprintf(stderr,"\t-x exclude # points [default 0]\n");
  fprintf(stderr,"\t-d delay  [default 1]\n");
  fprintf(stderr,"\t-M # of components, max. embedding dim. [default 1,10]\n");
  fprintf(stderr,"\t-c columns [default 1,...,# of components]\n");
  fprintf(stderr,"\t-t theiler-window [default 0]\n");
  fprintf(stderr,"\t-R max-epsilon "
	  "[default: max data interval]\n");
  fprintf(stderr,"\t-r min-epsilon [default: (max data interval)/1000]\n");
  fprintf(stderr,"\t-# #-of-epsilons [default 100]\n");
  fprintf(stderr,"\t-N max-#-of-pairs (0 means all) [default 1000]\n");
  fprintf(stderr,"\t-E use rescaled data [default: not rescaled]\n");
  fprintf(stderr," \t-o outfiles"
	  " [without exts.! default datafile[.d2][.h2][.stat][.c2]]\n");
  fprintf(stderr,"\t-V verbosity level [default: 1]\n\t\t"
          "0='only panic messages'\n\t\t"
          "1='+ input/output messages'\n\t\t"
	  "2='+ output message each time output is done\n");

  fprintf(stderr,"\t-h show these options\n");
  fprintf(stderr,"\n");
  exit(0);
}

void scan_options(int n,char **argv)
{
  char *out;

  if ((out=check_option(argv,n,'l','u')) != NULL)
    sscanf(out,"%lu",&length);
  if ((out=check_option(argv,n,'x','u')) != NULL)
    sscanf(out,"%lu",&exclude);
  if ((out=check_option(argv,n,'c','s')) != NULL)
    column=out;
  if ((out=check_option(argv,n,'d','u')) != NULL)
    sscanf(out,"%u",&DELAY);
  if ((out=check_option(argv,n,'M','2')) != NULL) {
    sscanf(out,"%u,%u",&DIM,&EMBED);
    dimset=1;
  }
  if ((out=check_option(argv,n,'t','u')) != NULL)
    sscanf(out,"%lu",&MINDIST);
  if ((out=check_option(argv,n,'R','f')) != NULL) {
    sscanf(out,"%lf",&EPSMAX);
    eps_max_set=1;
  }
  if ((out=check_option(argv,n,'r','f')) != NULL) {
    sscanf(out,"%lf",&EPSMIN);
    eps_min_set=1;
  }
  if ((out=check_option(argv,n,'#','u')) != NULL)
    sscanf(out,"%u",&HOWOFTEN);
  if ((out=check_option(argv,n,'N','u')) != NULL) {
    sscanf(out,"%lu",&MAXFOUND);
    if (MAXFOUND == 0)
      MAXFOUND=ULONG_MAX;
  }
  if ((out=check_option(argv,n,'E','n')) != NULL)
    rescale_set=1;
  if ((out=check_option(argv,n,'V','u')) != NULL)
    sscanf(out,"%u",&verbosity);
  if ((out=check_option(argv,n,'o','o')) != NULL)
    if (strlen(out) > 0)
      FOUT=out;
}

typedef struct {
  char *outd1,*outc1,*outh1,*outstat;
  time_t lasttime;
} DumpCtx;

/* Reproduces the original main()'s wall-clock-gated periodic dump of the
   .stat/.c2/.h2/.d2 files: written at most once every WHEN seconds, but
   always on the final center point (is_last), matching the original's
   "if (((time(&mytime)-lasttime) > WHEN) || (n == (nmax-1)) ||
   (imin > howoften1))". The per-row skip conditions the original expressed
   as fprintf() guards (norm[j]>0.0, found[.][j]>0.0, ...) are reproduced
   here as isnan() checks on the snapshot's NaN-gated c2/h2/d2 entries -
   see d2.h. */
static void dump_progress(const D2Result *snapshot, unsigned long centers_treated,
			   double current_eps_max, int is_last, void *user_data)
{
  DumpCtx *ctx=(DumpCtx*)user_data;
  time_t mytime;
  FILE *fout,*fstat;
  unsigned int i;
  unsigned long j;

  if (((time(&mytime)-ctx->lasttime) <= WHEN) && !is_last)
    return;
  ctx->lasttime=mytime;

  fstat=fopen(ctx->outstat,"w");
  if (verbosity&VER_USR1)
    fprintf(stderr,"Opened %s for writing\n",ctx->outstat);
  fprintf(fstat,"Center points treated so far= %ld\n",(long)centers_treated);
  fprintf(fstat,"Maximal epsilon in the moment= %e\n",current_eps_max);
  fclose(fstat);

  fout=fopen(ctx->outc1,"w");
  if (verbosity&VER_USR1)
    fprintf(stderr,"Opened %s for writing\n",ctx->outc1);
  fprintf(fout,"#center= %ld\n",(long)centers_treated);
  for (i=0;i<snapshot->n_blocks;i++) {
    fprintf(fout,"#dim= %u\n",i+1);
    for (j=0;j<snapshot->howoften;j++)
      if (!isnan(snapshot->c2[i][j]))
	fprintf(fout,"%e %e\n",snapshot->eps[j],snapshot->c2[i][j]);
    fprintf(fout,"\n\n");
  }
  fclose(fout);

  fout=fopen(ctx->outh1,"w");
  if (verbosity&VER_USR1)
    fprintf(stderr,"Opened %s for writing\n",ctx->outh1);
  fprintf(fout,"#center= %ld\n",(long)centers_treated);
  for (i=0;i<snapshot->n_blocks;i++) {
    fprintf(fout,"#dim= %u\n",i+1);
    for (j=0;j<snapshot->howoften;j++)
      if (!isnan(snapshot->h2[i][j]))
	fprintf(fout,"%e %e\n",snapshot->eps[j],snapshot->h2[i][j]);
    fprintf(fout,"\n\n");
  }
  fclose(fout);

  fout=fopen(ctx->outd1,"w");
  if (verbosity&VER_USR1)
    fprintf(stderr,"Opened %s for writing\n",ctx->outd1);
  fprintf(fout,"#center= %ld\n",(long)centers_treated);
  for (i=0;i<snapshot->n_blocks;i++) {
    fprintf(fout,"#dim= %u\n",i+1);
    for (j=1;j<snapshot->howoften;j++)
      if (!isnan(snapshot->d2[i][j]))
	fprintf(fout,"%e %e\n",snapshot->eps[j],snapshot->d2[i][j]);
    fprintf(fout,"\n\n");
  }
  fclose(fout);
}

int main(int argc,char **argv)
{
  char stdi=0;
  unsigned int i;
  double **series;
  DumpCtx ctx;
  D2Result *result;
  D2Error error;

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

  if (FOUT == NULL) {
    if (!stdi) {
      check_alloc(FOUT=calloc(strlen(infile)+1,(size_t)1));
      strcpy(FOUT,infile);
    }
    else {
      check_alloc(FOUT=calloc((size_t)6,(size_t)1));
      strcpy(FOUT,"stdin");
    }
  }
  if (column == NULL)
    series=(double**)get_multi_series(infile,&length,exclude,&DIM,"",dimset,
				      verbosity);
  else
    series=(double**)get_multi_series(infile,&length,exclude,&DIM,column,
				      dimset,verbosity);

  check_alloc(ctx.outd1=(char*)calloc(strlen(FOUT)+4,(size_t)1));
  check_alloc(ctx.outc1=(char*)calloc(strlen(FOUT)+4,(size_t)1));
  check_alloc(ctx.outh1=(char*)calloc(strlen(FOUT)+4,(size_t)1));
  check_alloc(ctx.outstat=(char*)calloc(strlen(FOUT)+6,(size_t)1));
  strcpy(ctx.outd1,FOUT);
  strcpy(ctx.outc1,FOUT);
  strcpy(ctx.outh1,FOUT);
  strcpy(ctx.outstat,FOUT);
  strcat(ctx.outd1,".d2");
  strcat(ctx.outc1,".c2");
  strcat(ctx.outh1,".h2");
  strcat(ctx.outstat,".stat");
  test_outfile(ctx.outd1);
  test_outfile(ctx.outc1);
  test_outfile(ctx.outh1);
  test_outfile(ctx.outstat);

  time(&ctx.lasttime);

  result=d2_compute((double *const *)series,length,DIM,EMBED,DELAY,MINDIST,
		     EPSMAX,eps_max_set,EPSMIN,eps_min_set,HOWOFTEN,MAXFOUND,
		     rescale_set,&error,dump_progress,&ctx);

  if (result == NULL) {
    if (error == D2_ERR_VECTOR_TOO_LARGE_FOR_LENGTH) {
      fprintf(stderr,"Embedding dimension and delay are too large.\n"
	      "The delay vector would be longer than the whole series."
	      " Exiting\n");
      exit(VECTOR_TOO_LARGE_FOR_LENGTH);
    }
    else {
      /* D2_ERR_RESCALE_ZERO_INTERVAL: reproduce rescale_data()'s own exit
	 message. d2_compute() already found this on a private copy of the
	 data; redo just the plain min/max scan (in the same component
	 order) here to name the same offending min/max in the message. */
      double min=0.0,interval=0.0;
      unsigned long k;
      for (i=0;i<DIM;i++) {
	min=interval=series[i][0];
	for (k=1;k<length;k++) {
	  if (series[i][k] < min) min=series[i][k];
	  if (series[i][k] > interval) interval=series[i][k];
	}
	interval -= min;
	if (interval == 0.0)
	  break;
      }
      fprintf(stderr,"rescale_data: data ranges from %e to %e. It makes\n"
	      "\t\tno sense to continue. Exiting!\n\n",min,min+interval);
      exit(RESCALE_DATA_ZERO_INTERVAL);
    }
  }

  d2_free(result);

  if (infile != NULL)
    free(infile);
  free(ctx.outd1);
  free(ctx.outh1);
  free(ctx.outc1);
  free(ctx.outstat);
  for (i=0;i<DIM;i++)
    free(series[i]);
  free(series);

  return 0;
}
