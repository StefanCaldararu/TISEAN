/* normal.c
 *
 * C replacement for TISEAN normal.f
 *
 * Provides:
 *   rms_
 *   normal_
 *   normal1_
 *   minmax_
 *
 * ABI-compatible with gfortran-generated symbols.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*------------------------------------------------------------------*/
/* rms_                                                             */
/*                                                                  */
/* return mean (sc) and rms amplitude (sd)                          */
/*------------------------------------------------------------------*/

void rms_(int *nmax, float *x, float *sc, float *sd)
{
    int n;
    float mean = 0.0f;
    float rmsdev = 0.0f;

    for (n = 0; n < *nmax; n++)
        mean += x[n];

    mean /= (float)(*nmax);

    for (n = 0; n < *nmax; n++) {
        float d = x[n] - mean;
        rmsdev += d * d;
    }

    rmsdev = sqrtf(rmsdev / (float)(*nmax));

    *sc = mean;
    *sd = rmsdev;
}

/*------------------------------------------------------------------*/
/* normal_                                                          */
/*                                                                  */
/* subtract mean                                                    */
/*------------------------------------------------------------------*/

void normal_(int *nmax, float *x, float *sc, float *sd)
{
    int n;

    rms_(nmax, x, sc, sd);

    for (n = 0; n < *nmax; n++)
        x[n] -= *sc;
}

/*------------------------------------------------------------------*/
/* normal1_                                                         */
/*                                                                  */
/* subtract mean and normalize to unit variance                     */
/*------------------------------------------------------------------*/

void normal1_(int *nmax, float *x, float *sc, float *sd)
{
    int n;

    rms_(nmax, x, sc, sd);

    if (*sd == 0.0f) {
        fprintf(stderr,
                "normal1: zero variance, cannot normalise\n");
        exit(EXIT_FAILURE);
    }

    for (n = 0; n < *nmax; n++)
        x[n] = (x[n] - *sc) / *sd;
}

/*------------------------------------------------------------------*/
/* minmax_                                                          */
/*                                                                  */
/* find minimum and maximum values                                  */
/*------------------------------------------------------------------*/

void minmax_(int *nmax, float *x, float *xmin, float *xmax)
{
    int n;

    *xmin = x[0];
    *xmax = x[0];

    for (n = 1; n < *nmax; n++) {

        if (x[n] < *xmin)
            *xmin = x[n];

        if (x[n] > *xmax)
            *xmax = x[n];
    }
}