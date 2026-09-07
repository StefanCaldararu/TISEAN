/* neigh.c
 *
 * C replacement for TISEAN neigh.f
 *
 * box-assisted neighbour search
 * see H. Kantz, T. Schreiber, Nonlinear Time Series Analysis, Cambridge
 * University Press (1997)
 * author T. Schreiber (1999), mneigh2 H. Kantz (2004)
 *
 * Provides:
 *   base_
 *   neigh_
 *   mbase_
 *   mneigh_
 *   mneigh2_
 *   mbase2_
 *
 * ABI-compatible with gfortran-generated symbols.
 *
 * y(nxx,mmax) (or y(nx,mdim)) is a Fortran two-dimensional array and is
 * stored column-major: y(n,is) is y[(is-1)*nxx + (n-1)] here, not
 * y[(n-1)*nxx + (is-1)]. jh(0:im*im) and jpntr(nmax) are supplied by the
 * caller; nothing is allocated in this file.
 */

#include <math.h>

#define IM 100
#define II 100000000

void base_(int *nmax, float *y, int *id, int *m, int *jh, int *jpntr,
           float *eps);
void neigh_(int *nmax, float *y, float *x, int *n, int *nlast, int *id,
            int *m, int *jh, int *jpntr, float *eps, int *nlist,
            int *nfound);
void mbase_(int *nmax, int *mmax, int *nxx, float *y, int *id, int *m,
            int *jh, int *jpntr, float *eps);
void mneigh_(int *nmax, int *mmax, int *nxx, float *y, int *n, int *nlast,
             int *id, int *m, int *jh, int *jpntr, float *eps, int *nlist,
             int *nfound);
void mneigh2_(int *nmax, int *mdim, float *y, int *nx, float *vx, int *jh,
              int *jpntr, float *eps, int *nlist, int *nfound);
void mbase2_(int *nmax, int *mmax, int *nxx, float *y, int *jh, int *jpntr,
             float *eps);

/* column-major index into a Fortran y(lead,*) array: y(n,is) */
static float y2(const float *y, int lead, int n, int is)
{
    return y[(is - 1) * lead + (n - 1)];
}

/*------------------------------------------------------------------*/
/* base_                                                             */
/*                                                                   */
/* build box histogram and pointer list, univariate series           */
/*------------------------------------------------------------------*/

void base_(int *nmax, float *y, int *id, int *m, int *jh, int *jpntr,
           float *eps)
{
    int nmax_v = *nmax, id_v = *id, m_v = *m;
    float eps_v = *eps;
    int n, i;

    for (i = 0; i <= IM * IM; i++)
        jh[i] = 0;

    for (n = (m_v - 1) * id_v + 1; n <= nmax_v; n++) { /* make histogram */
        i = ((int)(y[n - 1] / eps_v) + II) % IM;
        if (m_v > 1)
            i = IM * i
                + (((int)(y[n - 1 - (m_v - 1) * id_v] / eps_v) + II) % IM);
        jh[i]++;
    }
    for (i = 1; i <= IM * IM; i++) /* accumulate it */
        jh[i] += jh[i - 1];
    for (n = (m_v - 1) * id_v + 1; n <= nmax_v; n++) { /* fill list of pointers */
        i = ((int)(y[n - 1] / eps_v) + II) % IM;
        if (m_v > 1)
            i = IM * i
                + (((int)(y[n - 1 - (m_v - 1) * id_v] / eps_v) + II) % IM);
        jpntr[jh[i] - 1] = n;
        jh[i]--;
    }
}

/*------------------------------------------------------------------*/
/* neigh_                                                            */
/*                                                                   */
/* find neighbours of point n among points up to nlast, univariate   */
/*------------------------------------------------------------------*/

void neigh_(int *nmax, float *y, float *x, int *n, int *nlast, int *id,
            int *m, int *jh, int *jpntr, float *eps, int *nlist,
            int *nfound)
{
    int n_v = *n, nlast_v = *nlast, id_v = *id, m_v = *m;
    float eps_v = *eps;
    int nfound_v = 0;
    int kloop = (m_v == 1) ? 0 : 1;
    int jj, kk, j, k, jk, ip, np, i;

    (void)nmax;

    jj = (int)(y[n_v - 1] / eps_v);
    kk = (int)(y[n_v - 1 - (m_v - 1) * id_v] / eps_v);

    for (j = jj - 1; j <= jj + 1; j++) {         /* scan neighbouring boxes */
        for (k = kk - kloop; k <= kk + kloop; k++) {
            jk = (j + II) % IM;
            if (m_v > 1)
                jk = IM * jk + (k + II) % IM;
            for (ip = jh[jk + 1]; ip >= jh[jk] + 1; ip--) { /* time order */
                np = jpntr[ip - 1];
                if (np > nlast_v)
                    goto next_k;
                for (i = 0; i <= m_v - 1; i++) {
                    if (fabsf(y[n_v - 1 - i * id_v] - x[np - 1 - i * id_v])
                        >= eps_v)
                        goto next_ip;
                }
                nfound_v++;
                nlist[nfound_v - 1] = np;         /* make list of neighbours */
            next_ip:;
            }
        next_k:;
        }
    }
    *nfound = nfound_v;
}

/* versions for multivariate series, author T. Schreiber (1999) */

/*------------------------------------------------------------------*/
/* mbase_                                                            */
/*                                                                   */
/* build box histogram and pointer list, multivariate series         */
/*------------------------------------------------------------------*/

void mbase_(int *nmax, int *mmax, int *nxx, float *y, int *id, int *m,
            int *jh, int *jpntr, float *eps)
{
    int nmax_v = *nmax, mmax_v = *mmax, nxx_v = *nxx, id_v = *id, m_v = *m;
    float eps_v = *eps;
    int mt, n, i;

    if (mmax_v == 1) {
        base_(nmax, y, id, m, jh, jpntr, eps);
        return;
    }
    mt = (m_v - 1) / mmax_v + 1;

    for (i = 0; i <= IM * IM; i++)
        jh[i] = 0;

    for (n = (mt - 1) * id_v + 1; n <= nmax_v; n++) { /* make histogram */
        i = IM * (((int)(y2(y, nxx_v, n, 1) / eps_v) + II) % IM)
            + (((int)(y2(y, nxx_v, n, mmax_v) / eps_v) + II) % IM);
        jh[i]++;
    }
    for (i = 1; i <= IM * IM; i++) /* accumulate it */
        jh[i] += jh[i - 1];
    for (n = (mt - 1) * id_v + 1; n <= nmax_v; n++) { /* fill list of pointers */
        i = IM * (((int)(y2(y, nxx_v, n, 1) / eps_v) + II) % IM)
            + (((int)(y2(y, nxx_v, n, mmax_v) / eps_v) + II) % IM);
        jpntr[jh[i] - 1] = n;
        jh[i]--;
    }
}

/*------------------------------------------------------------------*/
/* mneigh_                                                           */
/*                                                                   */
/* find neighbours of point n among points up to nlast, multivariate */
/*------------------------------------------------------------------*/

void mneigh_(int *nmax, int *mmax, int *nxx, float *y, int *n, int *nlast,
             int *id, int *m, int *jh, int *jpntr, float *eps, int *nlist,
             int *nfound)
{
    int mmax_v = *mmax, nxx_v = *nxx;
    int n_v = *n, nlast_v = *nlast, id_v = *id, m_v = *m;
    float eps_v = *eps;
    int mt, nfound_v;
    int jj, kk, j, k, jk, ip, np, i, is, mcount;

    if (mmax_v == 1) {
        neigh_(nmax, y, y, n, nlast, id, m, jh, jpntr, eps, nlist, nfound);
        return;
    }
    mt = (m_v - 1) / mmax_v + 1;
    nfound_v = 0;
    jj = (int)(y2(y, nxx_v, n_v, 1) / eps_v);
    kk = (int)(y2(y, nxx_v, n_v, mmax_v) / eps_v);

    for (j = jj - 1; j <= jj + 1; j++) {         /* scan neighbouring boxes */
        for (k = kk - 1; k <= kk + 1; k++) {
            jk = IM * ((j + II) % IM) + (k + II) % IM;
            for (ip = jh[jk + 1]; ip >= jh[jk] + 1; ip--) { /* time order */
                np = jpntr[ip - 1];
                if (np > nlast_v)
                    goto next_k;
                mcount = 0;
                for (i = mt - 1; i >= 0; i--) {
                    for (is = 1; is <= mmax_v; is++) {
                        mcount++;
                        if (mcount > m_v)
                            goto matched;
                        if (fabsf(y2(y, nxx_v, n_v - i * id_v, is)
                                  - y2(y, nxx_v, np - i * id_v, is))
                            >= eps_v)
                            goto next_ip;
                    }
                }
            matched:
                nfound_v++;
                nlist[nfound_v - 1] = np;         /* make list of neighbours */
            next_ip:;
            }
        next_k:;
        }
    }
    *nfound = nfound_v;
}

/*>---------------------------------------------------------------------
 * modified version for multivariate series
 * author H. Kantz (2004)
 *
 * search neighbours for vx among the set of all y's
 * multivariate: mdim: spatial dimension
 * no additional delay!
 */

/*------------------------------------------------------------------*/
/* mneigh2_                                                          */
/*------------------------------------------------------------------*/

void mneigh2_(int *nmax, int *mdim, float *y, int *nx, float *vx, int *jh,
              int *jpntr, float *eps, int *nlist, int *nfound)
{
    int mdim_v = *mdim, nx_v = *nx;
    float eps_v = *eps;
    int nfound_v = 0;
    int jj, kk, j, k, jk, ip, np, is;

    (void)nmax;

    jj = (int)(vx[0] / eps_v);
    kk = (int)(vx[mdim_v - 1] / eps_v);

    for (j = jj - 1; j <= jj + 1; j++) {         /* scan neighbouring boxes */
        for (k = kk - 1; k <= kk + 1; k++) {
            jk = IM * ((j + II) % IM) + (k + II) % IM;
            for (ip = jh[jk + 1]; ip >= jh[jk] + 1; ip--) { /* time order */
                np = jpntr[ip - 1];
                for (is = 1; is <= mdim_v; is++) {
                    if (fabsf(vx[is - 1] - y2(y, nx_v, np, is)) >= eps_v)
                        goto next_ip;
                }
                nfound_v++;
                nlist[nfound_v - 1] = np;         /* make list of neighbours */
            next_ip:;
            }
        }
    }
    *nfound = nfound_v;
}

/*------------------------------------------------------------------*/
/* mbase2_                                                           */
/*                                                                   */
/* mbase2 has no callers anywhere in the tree. Its mmax.eq.1 branch  */
/* in the original Fortran calls base(nmax,y,id,m,jh,jpntr,eps) with  */
/* id and m -- neither of which is a parameter of mbase2 -- and loop  */
/* 40 below uses the same undefined id in (mmax-1)*id+1. That is a    */
/* latent bug in the source, not something introduced by this port:   */
/* both uninitialised variables are given the value 0 here rather     */
/* than guessed at, and since nothing calls this routine the path is  */
/* unreachable and its behaviour is unverifiable either way.          */
/*------------------------------------------------------------------*/

void mbase2_(int *nmax, int *mmax, int *nxx, float *y, int *jh, int *jpntr,
             float *eps)
{
    int nmax_v = *nmax, mmax_v = *mmax, nxx_v = *nxx;
    float eps_v = *eps;
    int id_v = 0, m_v = 0;
    int n, i;

    if (mmax_v == 1) {
        base_(nmax, y, &id_v, &m_v, jh, jpntr, eps);
        return;
    }

    for (i = 0; i <= IM * IM; i++)
        jh[i] = 0;

    for (n = 1; n <= nmax_v; n++) { /* make histogram */
        i = IM * (((int)(y2(y, nxx_v, n, 1) / eps_v) + II) % IM)
            + (((int)(y2(y, nxx_v, n, mmax_v) / eps_v) + II) % IM);
        jh[i]++;
    }
    for (i = 1; i <= IM * IM; i++) /* accumulate it */
        jh[i] += jh[i - 1];
    for (n = (mmax_v - 1) * id_v + 1; n <= nmax_v; n++) { /* fill list of pointers */
        i = IM * (((int)(y2(y, nxx_v, n, 1) / eps_v) + II) % IM)
            + (((int)(y2(y, nxx_v, n, mmax_v) / eps_v) + II) % IM);
        jpntr[jh[i] - 1] = n;
        jh[i]--;
    }
}
