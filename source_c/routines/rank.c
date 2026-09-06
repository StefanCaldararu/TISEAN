/* rank.c
 *
 * C replacement for TISEAN rank.f
 *
 * box assisted sorting/ranking utilities
 * author T. Schreiber (1998) based on earlier versions
 *
 * Provides:
 *   rank_
 *   indexx_
 *   rank2index_
 *   sort_
 *   rank2sort_
 *   index2sort_
 *   which_
 *
 * ABI-compatible with gfortran-generated symbols.
 *
 * All `list` entries are 1-based Fortran indices, even though this file
 * addresses `x` and `list` with 0-based C indexing ([i-1] for a 1-based
 * value i). Callers index their own arrays with the values stored here.
 */

#include <stdlib.h>

extern void minmax_(int *nmax, float *x, float *xmin, float *xmax);

void rank_(int *nmax, float *x, int *list);
void indexx_(int *nmax, float *x, int *list);
void rank2index_(int *nmax, int *list);
void sort_(int *nmax, float *x, int *list);
void rank2sort_(int *nmax, float *x, int *list);
void index2sort_(int *nmax, float *x, int *list);
float which_(int *nmax, float *x, int *k, int *list);

/*------------------------------------------------------------------*/
/* rank_                                                             */
/*                                                                   */
/* rank points in x                                                  */
/*------------------------------------------------------------------*/

void rank_(int *nmax, float *x, int *list)
{
    int nmax_v = *nmax;
    float xmin, xmax;
    int nptr = 100000;
    int nl, n, i;
    float sc;
    int *jptr;
    int rankn;

    minmax_(nmax, x, &xmin, &xmax);

    if (xmin == xmax) {
        for (n = 1; n <= nmax_v; n++)
            list[n - 1] = n;
        return;
    }

    nl = (nptr < nmax_v / 2) ? nptr : nmax_v / 2;
    sc = (float)(nl - 1) / (xmax - xmin);

    jptr = malloc((size_t)(nl + 1) * sizeof(int));

    for (i = 0; i <= nl; i++)
        jptr[i] = 0;

    for (n = 1; n <= nmax_v; n++) {
        float xn = x[n - 1];
        int idx = (int)((xn - xmin) * sc);
        int ip = jptr[idx];

        if (ip == 0 || xn <= x[ip - 1]) {
            jptr[idx] = n;
        } else {
            int ipp;
            do {
                ipp = ip;
                ip = list[ip - 1];
            } while (ip > 0 && xn > x[ip - 1]);
            list[ipp - 1] = n;
        }
        list[n - 1] = ip;
    }

    rankn = 0;
    for (i = 0; i <= nl; i++) {
        int ip = jptr[i];
        while (ip != 0) {
            int ipp = ip;
            rankn++;
            ip = list[ip - 1];
            list[ipp - 1] = rankn;
        }
    }

    free(jptr);
}

/*------------------------------------------------------------------*/
/* indexx_                                                           */
/*                                                                   */
/* make index table using rank                                       */
/*------------------------------------------------------------------*/

void indexx_(int *nmax, float *x, int *list)
{
    rank_(nmax, x, list);
    rank2index_(nmax, list);
}

/*------------------------------------------------------------------*/
/* rank2index_                                                       */
/*                                                                   */
/* converts a list of ranks into an index table (or vice versa)      */
/* in place                                                          */
/*------------------------------------------------------------------*/

void rank2index_(int *nmax, int *list)
{
    int nmax_v = *nmax;
    int n, i;

    for (i = 1; i <= nmax_v; i++)
        list[i - 1] = -list[i - 1];

    for (n = 1; n <= nmax_v; n++) {
        int ib, im, it;

        if (list[n - 1] > 0)
            continue; /* has been put in place already */

        ib = n;
        im = -list[n - 1];
        for (;;) {
            it = -list[im - 1];
            list[im - 1] = ib;
            if (it == n) {
                list[n - 1] = im;
                break;
            }
            ib = im;
            im = it;
        }
    }
}

/*------------------------------------------------------------------*/
/* sort_                                                             */
/*                                                                   */
/* sort using rank and rank2sort                                     */
/*------------------------------------------------------------------*/

void sort_(int *nmax, float *x, int *list)
{
    rank_(nmax, x, list);
    rank2sort_(nmax, x, list);
}

/*------------------------------------------------------------------*/
/* rank2sort_                                                        */
/*                                                                   */
/* sort x using list of ranks                                        */
/*------------------------------------------------------------------*/

void rank2sort_(int *nmax, float *x, int *list)
{
    int nmax_v = *nmax;
    int n, i;

    for (i = 1; i <= nmax_v; i++)
        list[i - 1] = -list[i - 1];

    for (n = 1; n <= nmax_v; n++) {
        int ib, it;
        float hb, ht;

        if (list[n - 1] > 0)
            continue; /* has been put in place already */

        ib = n;
        hb = x[n - 1];
        for (;;) {
            it = -list[ib - 1];
            list[ib - 1] = it;
            ht = x[it - 1];
            x[it - 1] = hb;
            if (it == n)
                break;
            ib = it;
            hb = ht;
        }
    }
}

/*------------------------------------------------------------------*/
/* index2sort_                                                       */
/*                                                                   */
/* sort x using list of indices                                      */
/*------------------------------------------------------------------*/

void index2sort_(int *nmax, float *x, int *list)
{
    int nmax_v = *nmax;
    int n, i;

    for (i = 1; i <= nmax_v; i++)
        list[i - 1] = -list[i - 1];

    for (n = 1; n <= nmax_v; n++) {
        int ib, it;
        float h;

        if (list[n - 1] > 0)
            continue; /* has been put in place already */

        ib = n;
        h = x[n - 1];
        for (;;) {
            it = -list[ib - 1];
            list[ib - 1] = it;
            if (it == n) {
                x[ib - 1] = h;
                break;
            }
            x[ib - 1] = x[it - 1];
            ib = it;
        }
    }
}

/*------------------------------------------------------------------*/
/* which_                                                            */
/*                                                                   */
/* find the k-th ranked value of x (clobbers list as scratch)        */
/*------------------------------------------------------------------*/

float which_(int *nmax, float *x, int *k, int *list)
{
    indexx_(nmax, x, list);
    return x[list[*k - 1] - 1];
}
