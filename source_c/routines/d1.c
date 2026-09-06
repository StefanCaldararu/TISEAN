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
/* d1.c
 *
 * C replacement for TISEAN d1.f
 *
 * d1 with finite sample correction following Grassberger, subroutine for c1
 *
 * Provides:
 *   d1_
 *   psi_
 *
 * ABI-compatible with gfortran-generated symbols.
 *
 * y(nxx,mmax) is a Fortran two-dimensional array and is stored column-major:
 * y(n,is) is y[(is-1)*nxx + (n-1)] here, not y[(n-1)*nxx + (is-1)].
 *
 * The reference-point shuffle uses source_c/routines/rand.c instead of
 * SLATEC's rand(), which the Fortran version relied on. The two generators
 * cannot be made to agree, so the numbers this produces differ from d1.f's;
 * see tests/test_c1.py for why that is safe (the mass column never depends
 * on which points are shuffled in, only the radius does, and that is
 * checked for agreement across seeds rather than for an exact value).
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define IM 100
#define NX 100000
#define TINY_VAL 1e-20f

extern void mbase_(int *nmax, int *mmax, int *nxx, float *y, int *id,
                    int *m, int *jh, int *jpntr, float *eps);
extern void mneigh_(int *nmax, int *mmax, int *nxx, float *y, int *n,
                     int *nlast, int *id, int *m, int *jh, int *jpntr,
                     float *eps, int *nlist, int *nfound);
extern float which_(int *nmax, float *x, int *k, int *list);
extern void rms_(int *nmax, float *x, float *sc, float *sd);
extern void rnd_init(unsigned long iseed);
extern unsigned long rnd_long(void);

void d1_(int *nmax, int *mmax, int *nxx, float *y, int *id, int *m,
         int *ncmin, float *pr, float *pln, float *eln, int *nmin,
         int *kmax);
float psi_(int *i);

/* column-major index into a Fortran y(lead,*) array: y(n,is) */
static float y2(const float *y, int lead, int n, int is)
{
    return y[(is - 1) * lead + (n - 1)];
}

/* -I no longer reaches this routine -- c1.f seeds the SLATEC generator,
   not this one -- so the shuffle is seeded from the wall clock instead. */
static float uniform01(void)
{
    static int seeded = 0;

    if (!seeded) {
        rnd_init((unsigned long)time(NULL));
        seeded = 1;
    }

    return (float)((double)rnd_long() / ((double)ULONG_MAX + 1.0));
}

/*------------------------------------------------------------------*/
/* d1_                                                               */
/*------------------------------------------------------------------*/

void d1_(int *nmax, int *mmax, int *nxx, float *y, int *id, int *m,
         int *ncmin, float *pr, float *pln, float *eln, int *nmin,
         int *kmax)
{
    static int jh[IM * IM + 1];
    static int ju[NX], jpntr[NX], nlist[NX], nwork[NX];
    static float d[NX];

    int mmax_v = *mmax, nxx_v = *nxx, id_v = *id, m_v = *m;
    int ncmin_v = *ncmin, nmin_v = *nmin, kmax_v = *kmax;
    int mt, ncomp, kpr, k, n0, ncomp_full;
    int i, iperm, ih, iu, nn, n, nf, ip, np, nmd, mcount, is, iunp, nfound;
    float sc, sd, eps, dis, e, eln_acc;

    if (*nmax > NX) {
        fprintf(stderr, "d1: make nx larger.\n");
        exit(EXIT_FAILURE);
    }

    mt = (m_v - 1) / mmax_v + 1;
    ncomp = *nmax - (mt - 1) * id_v;
    kpr = (int)(expf(*pr) * (float)(ncomp - 2 * nmin_v - 1)) + 1;
    k = (int)(expf(*pln) * (float)(ncomp - 2 * nmin_v - 1)) + 1;
    if (k > kmax_v) {
        ncomp = (int)((float)(ncomp - 2 * nmin_v - 1) * (float)kmax_v / (float)k
                      + (float)(2 * nmin_v + 1));
        k = kmax_v;
    }
    *pln = psi_(&k) - logf((float)(ncomp - 2 * nmin_v - 1));
    if (k == kpr)
        return;

    fprintf(stderr, "Mass %g : k= %d , N= %d\n", (double)expf(*pln), k, ncomp);

    rms_(nmax, y, &sc, &sd);
    eps = expf(*pln / (float)m_v) * sd;

    n0 = *nmax - (mt - 1) * id_v;
    for (i = 1; i <= n0; i++)
        ju[i - 1] = i + (mt - 1) * id_v;

    for (i = 1; i <= n0; i++) {
        iperm = (int)(uniform01() * (float)n0) + 1;
        if (iperm > n0)
            iperm = n0;
        ih = ju[i - 1];
        ju[i - 1] = ju[iperm - 1];
        ju[iperm - 1] = ih;
    }

    iu = ncmin_v;
    eln_acc = 0.0f;
    ncomp_full = ncomp + (mt - 1) * id_v;

    do {
        mbase_(&ncomp_full, mmax, nxx, y, id, m, jh, jpntr, &eps);
        iunp = 0;

        for (nn = 1; nn <= iu; nn++) {
            n = ju[nn - 1];
            mneigh_(nmax, mmax, nxx, y, &n, nmax, id, m, jh, jpntr, &eps,
                    nlist, &nfound);
            nf = 0;

            for (ip = 1; ip <= nfound; ip++) {
                np = nlist[ip - 1];
                nmd = abs(np - n) % ncomp;
                if (nmd <= nmin_v || nmd >= ncomp - nmin_v)
                    continue; /* temporal neighbours */

                nf++;
                dis = 0.0f;
                mcount = 0;
                for (i = mt - 1; i >= 0; i--) {
                    for (is = 1; is <= mmax_v; is++) {
                        float diff;

                        mcount++;
                        if (mcount > m_v)
                            goto dis_done;

                        diff = fabsf(y2(y, nxx_v, n - i * id_v, is)
                                     - y2(y, nxx_v, np - i * id_v, is));
                        if (diff > dis)
                            dis = diff;
                    }
                }
            dis_done:
                d[nf - 1] = dis;
            }

            if (nf < k) {
                iunp++; /* mark for next sweep */
                ju[iunp - 1] = n;
            } else {
                e = which_(&nf, d, &k, nwork);
                eln_acc += logf((e > TINY_VAL) ? e : TINY_VAL);
            }
        }

        iu = iunp;
        eps *= sqrtf(2.0f);
    } while (iunp != 0);

    *eln = eln_acc / (float)(ncmin_v - (mt - 1) * id_v);
}

/*------------------------------------------------------------------*/
/* psi_                                                              */
/*                                                                   */
/* digamma function                                                  */
/* Copyright (C) T. Schreiber (1998)                                 */
/*------------------------------------------------------------------*/

float psi_(int *i)
{
    static const float p[21] = {
        0.f,
        -0.57721566490f,  0.42278433509f,  0.92278433509f,  1.25611766843f,
         1.50611766843f,  1.70611766843f,  1.87278433509f,  2.01564147795f,
         2.14064147795f,  2.25175258906f,  2.35175258906f,  2.44266167997f,
         2.52599501330f,  2.60291809023f,  2.67434666166f,  2.74101332832f,
         2.80351332832f,  2.86233685773f,  2.91789241329f,  2.97052399224f
    };
    int i_v = *i;

    if (i_v <= 20)
        return p[i_v];
    else
        return logf((float)i_v) - 1.0f / (2.0f * (float)i_v);
}
