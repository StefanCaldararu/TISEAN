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
/*Author: Rainer Hegger. Last modified: Sep 5, 2004 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <ctype.h>
#include "routines/tsa.h"
#include "../include/nstat_z.h"

#define WID_STR "Tests for nonstationarity by means of the average\n\t\
forecast error for a zeroth order fit"


#ifndef _MATH_H
#include <math.h>
#endif

double *series;

char epsset=0,causalset=0;
char *infile=NULL;
char *outfile=NULL,stdo=1,centerset=0;
char *firstwindow,*secondwindow;
unsigned int COLUMN=1,pieces;
unsigned int verbosity=0xff;
int DIM=3,DELAY=1,MINN=30,STEP=1;
int firstoffset= -1,secondoffset= -1;
double EPS0=1.e-3,EPSF=1.2;
unsigned long LENGTH=ULONG_MAX,exclude=0,center,causal;

void show_options(char *progname)
{
  what_i_do(progname,WID_STR);
  fprintf(stderr," Usage: %s -# [other options]\n",progname);
  fprintf(stderr," Options:\n");
  fprintf(stderr,"Everything not being a valid option will be interpreted"
          " as a possible"
          " datafile.\nIf no datafile is given stdin is read. Just - also"
          " means stdin\n");
  fprintf(stderr,"\t-l # of data to use [default: whole file]\n");
  fprintf(stderr,"\t-x # of lines to be ignored [default: 0]\n");
  fprintf(stderr,"\t-c column to read [default: 1]\n");
  fprintf(stderr,"\t-m embedding dimension [default: 3]\n");
  fprintf(stderr,"\t-d delay [default: 1]\n");
  fprintf(stderr,"\t-# # of pieces [no default]\n");
  fprintf(stderr,"\t-1 which pieces for the first window "
	  "[default: 1-pieces]\n");
  fprintf(stderr,"\t-2 which pieces for the second window "
	  "[default: 1-pieces]\n");
  fprintf(stderr,"\t-n # of reference points in the window [default: all]\n");
  fprintf(stderr,"\t-k minimal number of neighbors for the fit "
	  "[default: 30]\n");
  fprintf(stderr,"\t-r neighborhoud size to start with "
	  "[default: (data interval)/1000]\n");
  fprintf(stderr,"\t-f factor to increase size [default: 1.2]\n");
  fprintf(stderr,"\t-s steps to forecast [default: 1]\n");
  fprintf(stderr,"\t-C width of causality window [default: steps]\n");
  fprintf(stderr,"\t-o output file [default: 'datafile.nsz',"
	  " without -o: stdout]\n");
  fprintf(stderr,"\t-V verbosity level [default: 1]\n\t\t"
          "0='only panic messages'\n\t\t"
          "1='+ input/output messages'\n");
  fprintf(stderr,"\t-h show these options\n");
  fprintf(stderr,"\n\t The -# option has to be set\n");
  exit(0);
}

void parse_minus(char *str,char *array,char *wopt)
{
  int cm=0,i,strl,n1,n2;
  
  strl=strlen(str);
  for (i=0;i<strl;i++)
    if (str[i] == '-')
      cm++;
  if (cm > 1) {
    fprintf(stderr,"Invalid string for the %s option! "
	    "Please consult the help-page\n",wopt);
    exit(NSTAT_Z__INVALID_STRING_FOR_OPTION);
  }
  if (cm == 0) {
    sscanf(str,"%d",&n1);
    n1--;
    if (n1 < 0) {
      fprintf(stderr,"Numbers in %s option must be larger than 0!\n",wopt);
      exit(NSTAT_Z__NOT_UNSIGNED_FOR_OPTION);
    }
    if (n1 >= pieces) {
      fprintf(stderr,"Numbers in %s option must be smaller than %u!\n",wopt,
	      pieces);
      exit(NSTAT_Z__TOO_LARGE_FOR_OPTION);
    }
    array[n1]=1;
  }
  else {
    sscanf(str,"%d-%d",&n1,&n2);
    n1--;
    n2--;
    if ((n1 < 0) || (n2 < 0)) {
      fprintf(stderr,"Numbers in %s option must be larger than 0!\n",wopt);
      exit(NSTAT_Z__NOT_UNSIGNED_FOR_OPTION);
    }
    if ((n1 >= pieces) || (n2 >= pieces)) {
      fprintf(stderr,"Numbers in %s option must be smaller than %u!\n",wopt,
	      pieces+1);
      exit(NSTAT_Z__TOO_LARGE_FOR_OPTION);
    }
    if (n2 < n1) {
      i=n1;
      n1=n2;
      n2=i;
    }
    for (i=n1;i<=n2;i++)
      array[i]=1;
  }
}

void parse_comma(char *str,char *array,char *wopt)
{
  unsigned int strl,i,cp=1,which,iwhich;
  char **hstr;

  strl=strlen(str);
  for (i=0;i<strl;i++)
    if (str[i] == ',')
      cp++;

  if (cp == 1) {
    parse_minus(str,array,wopt);
    return ;
  }
  
  check_alloc(hstr=(char**)malloc(sizeof(char*)*cp));
  for (i=0;i<cp;i++)
    check_alloc(hstr[i]=(char*)calloc(strl,1));
  
  which=iwhich=0;
  for (i=0;i<strl;i++) {
    if (str[i] != ',')
      hstr[which][iwhich++]=str[i];
    else {
      which++;
      iwhich=0;
    }
  }
  for (i=0;i<cp;i++) {
    if (hstr[i][0] == '\0') {
      fprintf(stderr,"Invalid string for the %s option! "
	      "Please consult the help-page\n",wopt);
      exit(NSTAT_Z__INVALID_STRING_FOR_OPTION);
    }
    if (!isdigit(hstr[i][strlen(hstr[i])-1])) {
      fprintf(stderr,"Invalid string for the %s option! "
	      "Please consult the help-page\n",wopt);
      exit(NSTAT_Z__INVALID_STRING_FOR_OPTION);
    }
    parse_minus(hstr[i],array,wopt);
  }
  for (i=0;i<cp;i++)
    free(hstr[i]);
  free(hstr);
}

void parse_out(char *str,char *array,char *which)
{
  unsigned int i;
  char test;

  for (i=0;i<pieces;i++)
    array[i]=0;
  
  for (i=0;i<strlen(str);i++) {
    test= (str[i] == '-') || (str[i] == ',') || isdigit(str[i]);
    if (!test) {
      fprintf(stderr,"Invalid string for the %s option! "
	      "Please consult the help-page\n",which);
      exit(NSTAT_Z__INVALID_STRING_FOR_OPTION);
    }
  }
  if (!isdigit(str[strlen(str)-1])) {
    fprintf(stderr,"Invalid string for the %s option! "
	    "Please consult the help-page\n",which);
    exit(NSTAT_Z__INVALID_STRING_FOR_OPTION);
  }
  parse_comma(str,array,which);
}

void parse_offset(char *str,int *iwhich,char *array,char *which) 
{
  int i,strl;
  
  if (str[0] != '+')
    return;
  strl=strlen(str);
  for (i=1;i<strl;i++)
    if (!isdigit(str[i])) {
      fprintf(stderr,"Invalid string for the %s option! "
	      "Please consult the help-page\n",which);
      exit(NSTAT_Z__INVALID_STRING_FOR_OPTION);
    }
  sscanf(str,"+%d",iwhich);
  for (i=0;i<pieces;i++)
    array[i]=0;
}
      
void scan_options(int n,char **in)
{
  unsigned int i;
  char *out,piecesset=0;

  if ((out=check_option(in,n,'l','u')) != NULL)
    sscanf(out,"%lu",&LENGTH);
  if ((out=check_option(in,n,'x','u')) != NULL)
    sscanf(out,"%lu",&exclude);
  if ((out=check_option(in,n,'c','u')) != NULL)
    sscanf(out,"%u",&COLUMN);
  if ((out=check_option(in,n,'m','u')) != NULL)
    sscanf(out,"%u",&DIM);
  if ((out=check_option(in,n,'d','u')) != NULL)
    sscanf(out,"%u",&DELAY);
  if ((out=check_option(in,n,'V','u')) != NULL)
    sscanf(out,"%u",&verbosity);
  if ((out=check_option(in,n,'#','u')) != NULL) {
    sscanf(out,"%u",&pieces);
    if (pieces < 1)
      pieces=1;
    piecesset=1;
    check_alloc(firstwindow=(char*)malloc(pieces));
    check_alloc(secondwindow=(char*)malloc(pieces));
    for (i=0;i<pieces;i++)
      firstwindow[i]=secondwindow[i]=1;
  }
  if (!piecesset) {
    fprintf(stderr,"\tThe -# option wasn't set. Please add it!\n");
    exit(NSTAT_Z__OPTION_NOT_SET);
  }
  if ((out=check_option(in,n,'1','s')) != NULL) {
    parse_offset(out,&firstoffset,firstwindow,"-1");
    if (firstoffset == -1)
      parse_out(out,firstwindow,"-1");
  }
  if ((out=check_option(in,n,'2','s')) != NULL) {
    parse_offset(out,&secondoffset,secondwindow,"-2");
    if (secondoffset == -1)
      parse_out(out,secondwindow,"-2");
  }
  if ((out=check_option(in,n,'n','u')) != NULL) {
    sscanf(out,"%lu",&center);
    centerset=1;
  }
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
    stdo=0;
    if (strlen(out) > 0)
      outfile=out;
  }
}

int main(int argc,char **argv)
{
  char stdi=0,sdone;
  long i,first;
  unsigned long ipair;
  double dmin,dinterval;
  NstatZ *result;
  NstatZError error;
  FILE *file=NULL;

  if (scan_help(argc,argv))
    show_options(argv[0]);
  
  scan_options(argc,argv);

  if (!causalset)
    causal=STEP;

#ifndef OMIT_WHAT_I_DO
  if (verbosity&VER_INPUT)
    what_i_do(argv[0],WID_STR);
#endif

  infile=search_datafile(argc,argv,&COLUMN,verbosity);
  if (infile == NULL)
    stdi=1;

  if (outfile == NULL) {
    if (!stdi) {
      check_alloc(outfile=(char*)calloc(strlen(infile)+5,(size_t)1));
      sprintf(outfile,"%s.nsz",infile);
    }
    else {
      check_alloc(outfile=(char*)calloc((size_t)10,(size_t)1));
      sprintf(outfile,"stdin.nsz");
    }
  }
  if (!stdo)
    test_outfile(outfile);

  series=(double*)get_series(infile,&LENGTH,exclude,COLUMN,verbosity);

  result=nstat_z_compute(series,LENGTH,pieces,firstwindow,secondwindow,
			  firstoffset,secondoffset,(unsigned int)DIM,
			  (unsigned int)DELAY,(unsigned int)MINN,
			  (unsigned long)STEP,causal,center,centerset,
			  EPS0,epsset,EPSF,&error);
  if (result == NULL) {
    if (error == NSTAT_Z_ERR_ZERO_INTERVAL) {
      /* Reproduce rescale_data()'s exact message. nstat_z_compute() rescales
	 a private copy and doesn't report the range, but it never touches
	 our own copy of series, so redo the same scan here. */
      dmin=dinterval=series[0];
      for (i=1;i<LENGTH;i++) {
	if (series[i] < dmin) dmin=series[i];
	if (series[i] > dinterval) dinterval=series[i];
      }
      dinterval -= dmin;
      fprintf(stderr,"rescale_data: data ranges from %e to %e. It makes\n"
	      "\t\tno sense to continue. Exiting!\n\n",dmin,dmin+dinterval);
      exit(RESCALE_DATA_ZERO_INTERVAL);
    }
    else if (error == NSTAT_Z_ERR_ZERO_VARIANCE) {
      fprintf(stderr,"Variance of the data is zero. Exiting!\n\n");
      exit(VARIANCE_VAR_EQ_ZERO);
    }
    else {
      fprintf(stderr,"You chose too many pieces and will never find enough"
	      " neighbors!\n");
      exit(NSTAT_Z__TOO_MANY_PIECES);
    }
  }

  free(firstwindow);
  free(secondwindow);

  if (stdo) {
    if (verbosity&VER_INPUT)
      fprintf(stderr,"Writing to stdout\n");
  }
  else {
    file=fopen(outfile,"w");
    if (verbosity&VER_INPUT)
      fprintf(stderr,"Opened %s for writing\n",outfile);
  }

  ipair=0;
  for (first=0;first<pieces;first++) {
    sdone=0;
    while ((ipair < result->n_pairs) && (result->first[ipair] == (unsigned int)first)) {
      sdone=1;
      if (stdo)
	fprintf(stdout,"%ld %ld %e\n",first+1,(long)result->second[ipair]+1,
		result->value[ipair]);
      else {
	fprintf(file,"%ld %ld %e\n",first+1,(long)result->second[ipair]+1,
		result->value[ipair]);
	fflush(file);
      }
      ipair++;
    }
    if (sdone) {
      if (stdo)
	fprintf(stdout,"\n");
      else
	fprintf(file,"\n");
    }
  }

  if (!stdo)
    fclose(file);

  if (outfile != NULL)
    free(outfile);
  nstat_z_free(result);
  free(series);

  return 0;
}

