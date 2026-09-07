/* snls1.c
 *
 * C replacement for TISEAN/SLATEC snls1.f and its subsidiary routines
 * ENORM, QRFAC, QRSOLV, LMPAR, FDJAC3 and RWUPDT.
 *
 * SNLS1 minimizes the sum of squares of M nonlinear functions in N
 * variables by a modification of the Levenberg-Marquardt algorithm
 * (Argonne MINPACK LMDER/LMDIF/LMSTR, combined).
 *
 * Provides (Fortran ABI, callable from gfortran-compiled code):
 *   snls1_
 *   enorm_
 *
 * All floating point is single precision (REAL in the original Fortran) --
 * do not widen to double, callers depend on the exact single precision
 * behaviour.
 *
 * FCN is a Fortran subroutine passed by name; gfortran passes a plain code
 * pointer and every argument FCN receives is by reference. QRFAC, QRSOLV,
 * LMPAR, FDJAC3 and RWUPDT are only ever called from within this file, so
 * they are kept as ordinary (non-Fortran-ABI) static helpers.
 *
 * Only the IOPT=1 (forward-difference Jacobian), NPRINT=0 path is
 * exercised by the only caller (upo.f, via peri). The IOPT=2/3 branches
 * below are translated as faithfully as possible, but the CHKDER-based
 * user-Jacobian verification that the original runs on the first
 * iteration for those modes is intentionally omitted (CHKDER is out of
 * scope for this port and is purely diagnostic -- it never changes FVEC,
 * FJAC or INFO). This means IOPT=2/3 are NOT bit-for-bit faithful in NFEV
 * counting on their first iteration, and are untested.
 */

#include <math.h>
#include <float.h>

typedef void (*fcn_t)(int *iflag, int *m, int *n, float *x, float *fvec,
                       float *fjac, int *ldfjac);

/*------------------------------------------------------------------*/
/* ENORM                                                              */
/*                                                                    */
/* Euclidean norm of an N-vector, computed to avoid spurious          */
/* overflow/underflow.                                                */
/*------------------------------------------------------------------*/

static float enorm(int n, const float *x)
{
    const float rdwarf = 3.834e-20f;
    const float rgiant = 1.304e19f;
    const float one = 1.0f, zero = 0.0f;
    float s1, s2, s3, x1max, x3max, floatn, agiant, xabs;
    int i;

    s1 = zero;
    s2 = zero;
    s3 = zero;
    x1max = zero;
    x3max = zero;
    floatn = (float) n;
    agiant = rgiant / floatn;

    for (i = 1; i <= n; i++) {
        xabs = fabsf(x[i - 1]);
        if (xabs > rdwarf && xabs < agiant) {
            s2 += xabs * xabs;
        } else if (xabs <= rdwarf) {
            if (xabs <= x3max) {
                if (xabs != zero) s3 += (xabs / x3max) * (xabs / x3max);
            } else {
                s3 = one + s3 * (x3max / xabs) * (x3max / xabs);
                x3max = xabs;
            }
        } else {
            if (xabs <= x1max) {
                s1 += (xabs / x1max) * (xabs / x1max);
            } else {
                s1 = one + s1 * (x1max / xabs) * (x1max / xabs);
                x1max = xabs;
            }
        }
    }

    if (s1 != zero)
        return x1max * sqrtf(s1 + (s2 / x1max) / x1max);
    if (s2 != zero) {
        if (s2 >= x3max)
            return sqrtf(s2 * (one + (x3max / s2) * (x3max * s3)));
        return sqrtf(x3max * ((s2 / x3max) + (x3max * s3)));
    }
    return x3max * sqrtf(s3);
}

float enorm_(int *n, float *x)
{
    return enorm(*n, x);
}

/*------------------------------------------------------------------*/
/* QRFAC                                                              */
/*                                                                    */
/* Householder QR factorization with optional column pivoting.       */
/*------------------------------------------------------------------*/

static void qrfac(int m, int n, float *a, int lda, int pivot, int *ipvt,
                   float *sigma, float *acnorm, float *wa)
{
    const float one = 1.0f, p05 = 5.0e-2f, zero = 0.0f;
    const float epsmch = FLT_EPSILON;
    int i, j, jp1, k, kmax, minmn;
    float ajnorm, sum, temp;

#define A(i, j) a[((j) - 1) * lda + ((i) - 1)]

    for (j = 1; j <= n; j++) {
        acnorm[j - 1] = enorm(m, &A(1, j));
        sigma[j - 1] = acnorm[j - 1];
        wa[j - 1] = sigma[j - 1];
        if (pivot) ipvt[j - 1] = j;
    }

    minmn = m < n ? m : n;
    for (j = 1; j <= minmn; j++) {
        if (pivot) {
            kmax = j;
            for (k = j; k <= n; k++)
                if (sigma[k - 1] > sigma[kmax - 1]) kmax = k;
            if (kmax != j) {
                for (i = 1; i <= m; i++) {
                    temp = A(i, j);
                    A(i, j) = A(i, kmax);
                    A(i, kmax) = temp;
                }
                sigma[kmax - 1] = sigma[j - 1];
                wa[kmax - 1] = wa[j - 1];
                k = ipvt[j - 1];
                ipvt[j - 1] = ipvt[kmax - 1];
                ipvt[kmax - 1] = k;
            }
        }

        ajnorm = enorm(m - j + 1, &A(j, j));
        if (ajnorm != zero) {
            if (A(j, j) < zero) ajnorm = -ajnorm;
            for (i = j; i <= m; i++) A(i, j) /= ajnorm;
            A(j, j) += one;

            jp1 = j + 1;
            if (n >= jp1) {
                for (k = jp1; k <= n; k++) {
                    sum = zero;
                    for (i = j; i <= m; i++) sum += A(i, j) * A(i, k);
                    temp = sum / A(j, j);
                    for (i = j; i <= m; i++) A(i, k) -= temp * A(i, j);
                    if (pivot && sigma[k - 1] != zero) {
                        temp = A(j, k) / sigma[k - 1];
                        sigma[k - 1] *= sqrtf(fmaxf(zero, one - temp * temp));
                        if (p05 * (sigma[k - 1] / wa[k - 1]) * (sigma[k - 1] / wa[k - 1]) <= epsmch) {
                            sigma[k - 1] = enorm(m - j, &A(jp1, k));
                            wa[k - 1] = sigma[k - 1];
                        }
                    }
                }
            }
        }
        sigma[j - 1] = -ajnorm;
    }
#undef A
}

/*------------------------------------------------------------------*/
/* QRSOLV                                                             */
/*                                                                    */
/* Complete the solution of A*X=B, D*X=0 in the least squares sense,  */
/* given the QR factorization of A.                                  */
/*------------------------------------------------------------------*/

static void qrsolv(int n, float *r, int ldr, const int *ipvt, float *diag,
                    float *qtb, float *x, float *sigma, float *wa)
{
    const float p5 = 5.0e-1f, p25 = 2.5e-1f, zero = 0.0f;
    int i, j, jp1, k, kp1, l, nsing;
    float cosv, cotan, qtbpj, sinv, sum, tanv, temp;

#define R(i, j) r[((j) - 1) * ldr + ((i) - 1)]

    for (j = 1; j <= n; j++) {
        for (i = j; i <= n; i++) R(i, j) = R(j, i);
        x[j - 1] = R(j, j);
        wa[j - 1] = qtb[j - 1];
    }

    for (j = 1; j <= n; j++) {
        l = ipvt[j - 1];
        if (diag[l - 1] != zero) {
            for (k = j; k <= n; k++) sigma[k - 1] = zero;
            sigma[j - 1] = diag[l - 1];

            qtbpj = zero;
            for (k = j; k <= n; k++) {
                if (sigma[k - 1] != zero) {
                    if (fabsf(R(k, k)) >= fabsf(sigma[k - 1])) {
                        tanv = sigma[k - 1] / R(k, k);
                        cosv = p5 / sqrtf(p25 + p25 * tanv * tanv);
                        sinv = cosv * tanv;
                    } else {
                        cotan = R(k, k) / sigma[k - 1];
                        sinv = p5 / sqrtf(p25 + p25 * cotan * cotan);
                        cosv = sinv * cotan;
                    }

                    R(k, k) = cosv * R(k, k) + sinv * sigma[k - 1];
                    temp = cosv * wa[k - 1] + sinv * qtbpj;
                    qtbpj = -sinv * wa[k - 1] + cosv * qtbpj;
                    wa[k - 1] = temp;

                    kp1 = k + 1;
                    if (n >= kp1) {
                        for (i = kp1; i <= n; i++) {
                            temp = cosv * R(i, k) + sinv * sigma[i - 1];
                            sigma[i - 1] = -sinv * R(i, k) + cosv * sigma[i - 1];
                            R(i, k) = temp;
                        }
                    }
                }
            }
        }
        sigma[j - 1] = R(j, j);
        R(j, j) = x[j - 1];
    }

    nsing = n;
    for (j = 1; j <= n; j++) {
        if (sigma[j - 1] == zero && nsing == n) nsing = j - 1;
        if (nsing < n) wa[j - 1] = zero;
    }
    if (nsing >= 1) {
        for (k = 1; k <= nsing; k++) {
            j = nsing - k + 1;
            sum = zero;
            jp1 = j + 1;
            if (nsing >= jp1)
                for (i = jp1; i <= nsing; i++) sum += R(i, j) * wa[i - 1];
            wa[j - 1] = (wa[j - 1] - sum) / sigma[j - 1];
        }
    }

    for (j = 1; j <= n; j++) {
        l = ipvt[j - 1];
        x[l - 1] = wa[j - 1];
    }
#undef R
}

/*------------------------------------------------------------------*/
/* LMPAR                                                              */
/*                                                                    */
/* Determine the Levenberg-Marquardt parameter.                      */
/*------------------------------------------------------------------*/

static void lmpar(int n, float *r, int ldr, const int *ipvt, float *diag,
                   float *qtb, float delta, float *par, float *x,
                   float *sigma, float *wa1, float *wa2)
{
    const float p1 = 1.0e-1f, p001 = 1.0e-3f, zero = 0.0f;
    const float dwarf = FLT_MIN;
    int i, iter, j, jm1, jp1, k, l, nsing;
    float dxnorm, fp, gnorm, parc, parl, paru, sum, temp;

#define R(i, j) r[((j) - 1) * ldr + ((i) - 1)]

    nsing = n;
    for (j = 1; j <= n; j++) {
        wa1[j - 1] = qtb[j - 1];
        if (R(j, j) == zero && nsing == n) nsing = j - 1;
        if (nsing < n) wa1[j - 1] = zero;
    }
    if (nsing >= 1) {
        for (k = 1; k <= nsing; k++) {
            j = nsing - k + 1;
            wa1[j - 1] = wa1[j - 1] / R(j, j);
            temp = wa1[j - 1];
            jm1 = j - 1;
            if (jm1 >= 1)
                for (i = 1; i <= jm1; i++) wa1[i - 1] -= R(i, j) * temp;
        }
    }
    for (j = 1; j <= n; j++) {
        l = ipvt[j - 1];
        x[l - 1] = wa1[j - 1];
    }

    iter = 0;
    for (j = 1; j <= n; j++) wa2[j - 1] = diag[j - 1] * x[j - 1];
    dxnorm = enorm(n, wa2);
    fp = dxnorm - delta;
    if (fp > p1 * delta) {
        parl = zero;
        if (nsing >= n) {
            for (j = 1; j <= n; j++) {
                l = ipvt[j - 1];
                wa1[j - 1] = diag[l - 1] * (wa2[l - 1] / dxnorm);
            }
            for (j = 1; j <= n; j++) {
                sum = zero;
                jm1 = j - 1;
                if (jm1 >= 1)
                    for (i = 1; i <= jm1; i++) sum += R(i, j) * wa1[i - 1];
                wa1[j - 1] = (wa1[j - 1] - sum) / R(j, j);
            }
            temp = enorm(n, wa1);
            parl = ((fp / delta) / temp) / temp;
        }

        for (j = 1; j <= n; j++) {
            sum = zero;
            for (i = 1; i <= j; i++) sum += R(i, j) * qtb[i - 1];
            l = ipvt[j - 1];
            wa1[j - 1] = sum / diag[l - 1];
        }
        gnorm = enorm(n, wa1);
        paru = gnorm / delta;
        if (paru == zero) paru = dwarf / fminf(delta, p1);

        *par = fmaxf(*par, parl);
        *par = fminf(*par, paru);
        if (*par == zero) *par = gnorm / dxnorm;

        for (;;) {
            iter = iter + 1;

            if (*par == zero) *par = fmaxf(dwarf, p001 * paru);
            temp = sqrtf(*par);
            for (j = 1; j <= n; j++) wa1[j - 1] = temp * diag[j - 1];
            qrsolv(n, r, ldr, ipvt, wa1, qtb, x, sigma, wa2);
            for (j = 1; j <= n; j++) wa2[j - 1] = diag[j - 1] * x[j - 1];
            dxnorm = enorm(n, wa2);
            temp = fp;
            fp = dxnorm - delta;

            if (fabsf(fp) <= p1 * delta ||
                (parl == zero && fp <= temp && temp < zero) ||
                iter == 10)
                break;

            for (j = 1; j <= n; j++) {
                l = ipvt[j - 1];
                wa1[j - 1] = diag[l - 1] * (wa2[l - 1] / dxnorm);
            }
            for (j = 1; j <= n; j++) {
                wa1[j - 1] = wa1[j - 1] / sigma[j - 1];
                temp = wa1[j - 1];
                jp1 = j + 1;
                if (n >= jp1)
                    for (i = jp1; i <= n; i++) wa1[i - 1] -= R(i, j) * temp;
            }
            temp = enorm(n, wa1);
            parc = ((fp / delta) / temp) / temp;

            if (fp > zero) parl = fmaxf(parl, *par);
            if (fp < zero) paru = fminf(paru, *par);

            *par = fmaxf(parl, *par + parc);
        }
    }

    if (iter == 0) *par = zero;
#undef R
}

/*------------------------------------------------------------------*/
/* FDJAC3                                                             */
/*                                                                    */
/* Forward-difference approximation to the M by N Jacobian.          */
/*------------------------------------------------------------------*/

static void fdjac3(fcn_t fcn, int m, int n, float *x, const float *fvec,
                    float *fjac, int ldfjac, int *iflag, float epsfcn,
                    float *wa)
{
    const float zero = 0.0f;
    const float epsmch = FLT_EPSILON;
    float eps, h, temp;
    int i, j;

#define FJ(i, j) fjac[((j) - 1) * ldfjac + ((i) - 1)]

    eps = sqrtf(fmaxf(epsfcn, epsmch));
    *iflag = 1;
    for (j = 1; j <= n; j++) {
        temp = x[j - 1];
        h = eps * fabsf(temp);
        if (h == zero) h = eps;
        x[j - 1] = temp + h;
        fcn(iflag, &m, &n, x, wa, fjac, &ldfjac);
        if (*iflag < 0) break;
        x[j - 1] = temp;
        for (i = 1; i <= m; i++) FJ(i, j) = (wa[i - 1] - fvec[i - 1]) / h;
    }
#undef FJ
}

/*------------------------------------------------------------------*/
/* RWUPDT                                                             */
/*                                                                    */
/* QR decomposition update when a row is added to an upper           */
/* triangular matrix R.                                              */
/*------------------------------------------------------------------*/

static void rwupdt(int n, float *r, int ldr, const float *w, float *b,
                    float *alpha, float *cosv, float *sinv)
{
    const float one = 1.0f, p5 = 5.0e-1f, p25 = 2.5e-1f, zero = 0.0f;
    int i, j, jm1;
    float cotan, rowj, tanv, temp;

#define R(i, j) r[((j) - 1) * ldr + ((i) - 1)]

    for (j = 1; j <= n; j++) {
        rowj = w[j - 1];
        jm1 = j - 1;

        if (jm1 >= 1) {
            for (i = 1; i <= jm1; i++) {
                temp = cosv[i - 1] * R(i, j) + sinv[i - 1] * rowj;
                rowj = -sinv[i - 1] * R(i, j) + cosv[i - 1] * rowj;
                R(i, j) = temp;
            }
        }

        cosv[j - 1] = one;
        sinv[j - 1] = zero;
        if (rowj != zero) {
            if (fabsf(R(j, j)) >= fabsf(rowj)) {
                tanv = rowj / R(j, j);
                cosv[j - 1] = p5 / sqrtf(p25 + p25 * tanv * tanv);
                sinv[j - 1] = cosv[j - 1] * tanv;
            } else {
                cotan = R(j, j) / rowj;
                sinv[j - 1] = p5 / sqrtf(p25 + p25 * cotan * cotan);
                cosv[j - 1] = sinv[j - 1] * cotan;
            }

            R(j, j) = cosv[j - 1] * R(j, j) + sinv[j - 1] * rowj;
            temp = cosv[j - 1] * b[j - 1] + sinv[j - 1] * (*alpha);
            *alpha = -sinv[j - 1] * b[j - 1] + cosv[j - 1] * (*alpha);
            b[j - 1] = temp;
        }
    }
#undef R
}

/*------------------------------------------------------------------*/
/* SNLS1                                                              */
/*                                                                    */
/* Minimize the sum of the squares of M nonlinear functions in N     */
/* variables by a modification of the Levenberg-Marquardt algorithm. */
/*------------------------------------------------------------------*/

void snls1_(fcn_t fcn, int *iopt_p, int *m_p, int *n_p, float *x,
            float *fvec, float *fjac, int *ldfjac_p, float *ftol_p,
            float *xtol_p, float *gtol_p, int *maxfev_p, float *epsfcn_p,
            float *diag, int *mode_p, float *factor_p, int *nprint_p,
            int *info, int *nfev, int *njev, int *ipvt, float *qtf,
            float *wa1, float *wa2, float *wa3, float *wa4)
{
    const float one = 1.0f, p1 = 1.0e-1f, p5 = 5.0e-1f, p25 = 2.5e-1f,
                p75 = 7.5e-1f, p0001 = 1.0e-4f, zero = 0.0f;
    const float epsmch = FLT_EPSILON;

    int IOPT = *iopt_p, M = *m_p, N = *n_p, LDFJAC = *ldfjac_p;
    float FTOL = *ftol_p, XTOL = *xtol_p, GTOL = *gtol_p;
    int MAXFEV = *maxfev_p;
    float EPSFCN = *epsfcn_p;
    int MODE = *mode_p;
    float FACTOR = *factor_p;
    int NPRINT = *nprint_p;

    int i, iflag, ijunk = 1, iter, j, l, nrow, sing;
    float actred, delta, dirder, fnorm, fnorm1, gnorm, par, pnorm, prered,
          ratio, sum, temp, temp1, temp2, xnorm;

#define FJ(i, j) fjac[((j) - 1) * LDFJAC + ((i) - 1)]

    *info = 0;
    iflag = 0;
    *nfev = 0;
    *njev = 0;

    if (IOPT < 1 || IOPT > 3 || N <= 0 || M < N || LDFJAC < N ||
        FTOL < zero || XTOL < zero || GTOL < zero || MAXFEV <= 0 ||
        FACTOR <= zero)
        goto L300;
    if (IOPT < 3 && LDFJAC < M) goto L300;
    if (MODE == 2) {
        for (j = 1; j <= N; j++)
            if (diag[j - 1] <= zero) goto L300;
    }

    iflag = 1;
    ijunk = 1;
    fcn(&iflag, &M, &N, x, fvec, fjac, &ijunk);
    *nfev = 1;
    if (iflag < 0) goto L300;
    fnorm = enorm(M, fvec);

    par = zero;
    iter = 1;

    for (;;) {
        if (NPRINT > 0) {
            iflag = 0;
            if ((iter - 1) % NPRINT == 0)
                fcn(&iflag, &M, &N, x, fvec, fjac, &ijunk);
            if (iflag < 0) goto L300;
        }

        if (IOPT == 3) goto L475;
        if (IOPT != 1) {
            /* IOPT == 2: user supplies the full Jacobian. */
            iflag = 2;
            fcn(&iflag, &M, &N, x, fvec, fjac, &LDFJAC);
            (*njev)++;
            /* CHKDER-based derivative check on the first iteration is
               intentionally not ported -- see file header. */
            goto L420;
        }
        {
            iflag = 1;
            fdjac3(fcn, M, N, x, fvec, fjac, LDFJAC, &iflag, EPSFCN, wa4);
            *nfev += N;
        }
    L420:
        if (iflag < 0) goto L300;

        qrfac(M, N, fjac, LDFJAC, 1, ipvt, wa1, wa2, wa3);
        for (i = 1; i <= M; i++) wa4[i - 1] = fvec[i - 1];
        for (j = 1; j <= N; j++) {
            if (FJ(j, j) != zero) {
                sum = zero;
                for (i = j; i <= M; i++) sum += FJ(i, j) * wa4[i - 1];
                temp = -sum / FJ(j, j);
                for (i = j; i <= M; i++) wa4[i - 1] += FJ(i, j) * temp;
            }
            FJ(j, j) = wa1[j - 1];
            qtf[j - 1] = wa4[j - 1];
        }
        goto L560;

    L475:
        for (j = 1; j <= N; j++) {
            qtf[j - 1] = zero;
            for (i = 1; i <= N; i++) FJ(i, j) = zero;
        }
        for (i = 1; i <= M; i++) {
            nrow = i;
            iflag = 3;
            fcn(&iflag, &M, &N, x, fvec, wa3, &nrow);
            if (iflag < 0) goto L300;
            /* CHKDER-based derivative check on the first iteration is
               intentionally not ported -- see file header. */
            temp = fvec[i - 1];
            rwupdt(N, fjac, LDFJAC, wa3, qtf, &temp, wa1, wa2);
        }
        (*njev)++;

        sing = 0;
        for (j = 1; j <= N; j++) {
            if (FJ(j, j) == zero) sing = 1;
            ipvt[j - 1] = j;
            wa2[j - 1] = enorm(j, &FJ(1, j));
        }
        if (sing) {
            qrfac(N, N, fjac, LDFJAC, 1, ipvt, wa1, wa2, wa3);
            for (j = 1; j <= N; j++) {
                if (FJ(j, j) != zero) {
                    sum = zero;
                    for (i = j; i <= N; i++) sum += FJ(i, j) * qtf[i - 1];
                    temp = -sum / FJ(j, j);
                    for (i = j; i <= N; i++) qtf[i - 1] += FJ(i, j) * temp;
                }
                FJ(j, j) = wa1[j - 1];
            }
        }

    L560:
        if (iter == 1) {
            if (MODE != 2) {
                for (j = 1; j <= N; j++) {
                    diag[j - 1] = wa2[j - 1];
                    if (wa2[j - 1] == zero) diag[j - 1] = one;
                }
            }
            for (j = 1; j <= N; j++) wa3[j - 1] = diag[j - 1] * x[j - 1];
            xnorm = enorm(N, wa3);
            delta = FACTOR * xnorm;
            if (delta == zero) delta = FACTOR;
        }

        gnorm = zero;
        if (fnorm != zero) {
            for (j = 1; j <= N; j++) {
                l = ipvt[j - 1];
                if (wa2[l - 1] != zero) {
                    sum = zero;
                    for (i = 1; i <= j; i++) sum += FJ(i, j) * (qtf[i - 1] / fnorm);
                    gnorm = fmaxf(gnorm, fabsf(sum / wa2[l - 1]));
                }
            }
        }

        if (gnorm <= GTOL) *info = 4;
        if (*info != 0) goto L300;

        if (MODE != 2) {
            for (j = 1; j <= N; j++) diag[j - 1] = fmaxf(diag[j - 1], wa2[j - 1]);
        }

        do {
            lmpar(N, fjac, LDFJAC, ipvt, diag, qtf, delta, &par, wa1, wa2,
                  wa3, wa4);

            for (j = 1; j <= N; j++) {
                wa1[j - 1] = -wa1[j - 1];
                wa2[j - 1] = x[j - 1] + wa1[j - 1];
                wa3[j - 1] = diag[j - 1] * wa1[j - 1];
            }
            pnorm = enorm(N, wa3);

            if (iter == 1) delta = fminf(delta, pnorm);

            iflag = 1;
            fcn(&iflag, &M, &N, wa2, wa4, fjac, &ijunk);
            (*nfev)++;
            if (iflag < 0) goto L300;
            fnorm1 = enorm(M, wa4);

            actred = -one;
            if (p1 * fnorm1 < fnorm)
                actred = one - (fnorm1 / fnorm) * (fnorm1 / fnorm);

            for (j = 1; j <= N; j++) {
                wa3[j - 1] = zero;
                l = ipvt[j - 1];
                temp = wa1[l - 1];
                for (i = 1; i <= j; i++) wa3[i - 1] += FJ(i, j) * temp;
            }
            temp1 = enorm(N, wa3) / fnorm;
            temp2 = (sqrtf(par) * pnorm) / fnorm;
            prered = temp1 * temp1 + temp2 * temp2 / p5;
            dirder = -(temp1 * temp1 + temp2 * temp2);

            ratio = zero;
            if (prered != zero) ratio = actred / prered;

            if (ratio <= p25) {
                if (actred >= zero) temp = p5;
                else temp = p5 * dirder / (dirder + p5 * actred);
                if (p1 * fnorm1 >= fnorm || temp < p1) temp = p1;
                delta = temp * fminf(delta, pnorm / p1);
                par = par / temp;
            } else {
                if (!(par != zero && ratio < p75)) {
                    delta = pnorm / p5;
                    par = p5 * par;
                }
            }

            if (ratio >= p0001) {
                for (j = 1; j <= N; j++) {
                    x[j - 1] = wa2[j - 1];
                    wa2[j - 1] = diag[j - 1] * x[j - 1];
                }
                for (i = 1; i <= M; i++) fvec[i - 1] = wa4[i - 1];
                xnorm = enorm(N, wa2);
                fnorm = fnorm1;
                iter = iter + 1;
            }

            if (fabsf(actred) <= FTOL && prered <= FTOL && p5 * ratio <= one)
                *info = 1;
            if (delta <= XTOL * xnorm) *info = 2;
            if (fabsf(actred) <= FTOL && prered <= FTOL && p5 * ratio <= one &&
                *info == 2)
                *info = 3;
            if (*info != 0) goto L300;

            if (*nfev >= MAXFEV) *info = 5;
            if (fabsf(actred) <= epsmch && prered <= epsmch && p5 * ratio <= one)
                *info = 6;
            if (delta <= epsmch * xnorm) *info = 7;
            if (gnorm <= epsmch) *info = 8;
            if (*info != 0) goto L300;
        } while (ratio < p0001);
    }

L300:
    if (iflag < 0) *info = iflag;
    iflag = 0;
    if (NPRINT > 0) fcn(&iflag, &M, &N, x, fvec, fjac, &ijunk);
    /* XERMSG diagnostic reporting for the various INFO/IFLAG outcomes is
       intentionally not ported -- see file header; it only ever prints,
       it never changes INFO or any array contents. */
#undef FJ
}
