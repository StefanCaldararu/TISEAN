/* rs.c
 *
 * C replacement for SLATEC/EISPACK rs.f (and the tred1/tred2/tql2/tqlrat/
 * pythag helpers it calls), ported directly from the single-precision
 * Fortran sources rather than adapted from the double-precision C eigen()
 * in eigen.c -- rs.f is REAL (single precision) throughout, and widening
 * to double and rounding back does not reproduce the same bits (nor,
 * for eigenvectors, even the same signs) as running in single precision
 * throughout.
 *
 * Provides:
 *   rs_
 *
 * ABI-compatible with gfortran-generated symbols. a and z are Fortran
 * two-dimensional arrays (column-major); nm is their declared leading
 * dimension, n the active order -- nm and n are not interchangeable.
 */

#include <math.h>
#include <float.h>

/* sqrt(a*a + b*b) without destructive overflow/underflow. */
static float pythag_s(float a, float b)
{
    float p, q, r, s, t;

    p = fmaxf(fabsf(a), fabsf(b));
    q = fminf(fabsf(a), fabsf(b));
    if (q == 0.0f)
        return p;

    for (;;) {
        r = (q / p) * (q / p);
        t = 4.0f + r;
        if (t == 4.0f)
            break;
        s = r / t;
        p += 2.0f * p * s;
        q *= s;
    }
    return p;
}

/* Reduce a real symmetric matrix to tridiagonal form (eigenvalues only path). */
static void tred1_s(int *nm, int *n, float *a, float *d, float *e, float *e2)
{
    int NM = *nm, N = *n;
    int i, j, k, l, ii, jp1;
    float f, g, h, scale;

#define A(i, j) a[((j) - 1) * NM + ((i) - 1)]
#define D(i) d[(i) - 1]
#define E(i) e[(i) - 1]
#define E2(i) e2[(i) - 1]

    for (i = 1; i <= N; i++)
        D(i) = A(i, i);

    for (ii = 1; ii <= N; ii++) {
        i = N + 1 - ii;
        l = i - 1;
        h = 0.0f;
        scale = 0.0f;
        if (l < 1) {
            E(i) = 0.0f;
            E2(i) = 0.0f;
            goto L290;
        }
        for (k = 1; k <= l; k++)
            scale += fabsf(A(i, k));

        if (scale == 0.0f) {
            E(i) = 0.0f;
            E2(i) = 0.0f;
            goto L290;
        }

        for (k = 1; k <= l; k++) {
            A(i, k) /= scale;
            h += A(i, k) * A(i, k);
        }

        E2(i) = scale * scale * h;
        f = A(i, l);
        g = -copysignf(sqrtf(h), f);
        E(i) = scale * g;
        h -= f * g;
        A(i, l) = f - g;
        if (l == 1)
            goto L270;
        f = 0.0f;

        for (j = 1; j <= l; j++) {
            g = 0.0f;
            for (k = 1; k <= j; k++)
                g += A(j, k) * A(i, k);
            jp1 = j + 1;
            if (l >= jp1) {
                for (k = jp1; k <= l; k++)
                    g += A(k, j) * A(i, k);
            }
            E(j) = g / h;
            f += E(j) * A(i, j);
        }

        h = f / (h + h);
        for (j = 1; j <= l; j++) {
            f = A(i, j);
            g = E(j) - h * f;
            E(j) = g;
            for (k = 1; k <= j; k++)
                A(j, k) = A(j, k) - f * E(k) - g * A(i, k);
        }

L270:
        for (k = 1; k <= l; k++)
            A(i, k) *= scale;

L290:
        h = D(i);
        D(i) = A(i, i);
        A(i, i) = h;
    }

#undef A
#undef D
#undef E
#undef E2
}

/* Reduce a real symmetric matrix to tridiagonal form, accumulating the
 * orthogonal transformation (eigenvalues + eigenvectors path). */
static void tred2_s(int *nm, int *n, float *a, float *d, float *e, float *z)
{
    int NM = *nm, N = *n;
    int i, j, k, l, ii, jp1;
    float f, g, h, hh, scale;

#define A(i, j) a[((j) - 1) * NM + ((i) - 1)]
#define Z(i, j) z[((j) - 1) * NM + ((i) - 1)]
#define D(i) d[(i) - 1]
#define E(i) e[(i) - 1]

    for (i = 1; i <= N; i++)
        for (j = 1; j <= i; j++)
            Z(i, j) = A(i, j);

    if (N == 1)
        goto L320;

    for (ii = 2; ii <= N; ii++) {
        i = N + 2 - ii;
        l = i - 1;
        h = 0.0f;
        scale = 0.0f;
        if (l >= 2) {
            for (k = 1; k <= l; k++)
                scale += fabsf(Z(i, k));
        }

        if (l < 2 || scale == 0.0f) {
            E(i) = Z(i, l);
            goto L290;
        }

        for (k = 1; k <= l; k++) {
            Z(i, k) /= scale;
            h += Z(i, k) * Z(i, k);
        }

        f = Z(i, l);
        g = -copysignf(sqrtf(h), f);
        E(i) = scale * g;
        h -= f * g;
        Z(i, l) = f - g;
        f = 0.0f;

        for (j = 1; j <= l; j++) {
            Z(j, i) = Z(i, j) / h;
            g = 0.0f;
            for (k = 1; k <= j; k++)
                g += Z(j, k) * Z(i, k);
            jp1 = j + 1;
            if (l >= jp1) {
                for (k = jp1; k <= l; k++)
                    g += Z(k, j) * Z(i, k);
            }
            E(j) = g / h;
            f += E(j) * Z(i, j);
        }

        hh = f / (h + h);
        for (j = 1; j <= l; j++) {
            f = Z(i, j);
            g = E(j) - hh * f;
            E(j) = g;
            for (k = 1; k <= j; k++)
                Z(j, k) = Z(j, k) - f * E(k) - g * Z(i, k);
        }

L290:
        D(i) = h;
    }

L320:
    D(1) = 0.0f;
    E(1) = 0.0f;

    for (i = 1; i <= N; i++) {
        l = i - 1;
        if (D(i) != 0.0f) {
            for (j = 1; j <= l; j++) {
                g = 0.0f;
                for (k = 1; k <= l; k++)
                    g += Z(i, k) * Z(k, j);
                for (k = 1; k <= l; k++)
                    Z(k, j) -= g * Z(k, i);
            }
        }

        D(i) = Z(i, i);
        Z(i, i) = 1.0f;
        if (l < 1)
            continue;
        for (j = 1; j <= l; j++) {
            Z(i, j) = 0.0f;
            Z(j, i) = 0.0f;
        }
    }

#undef A
#undef Z
#undef D
#undef E
}

/* Eigenvalues and eigenvectors of a symmetric tridiagonal matrix by the QL
 * method (also finishes the eigenvectors of the full matrix, if z was
 * primed with the tred2_s transformation matrix). */
static void tql2_s(int *nm, int *n, float *d, float *e, float *z, int *ierr)
{
    int NM = *nm, N = *n;
    int i, j, k, l, m, ii, l1, l2, mml;
    float b, c, c2, c3, dl1, el1, f, g, h, p, r, s, s2;

#define Z(i, j) z[((j) - 1) * NM + ((i) - 1)]
#define D(i) d[(i) - 1]
#define E(i) e[(i) - 1]

    *ierr = 0;
    if (N == 1)
        goto L1001;

    for (i = 2; i <= N; i++)
        E(i - 1) = E(i);

    f = 0.0f;
    b = 0.0f;
    E(N) = 0.0f;

    for (l = 1; l <= N; l++) {
        j = 0;
        h = fabsf(D(l)) + fabsf(E(l));
        if (b < h)
            b = h;

        for (m = l; m <= N; m++) {
            if (b + fabsf(E(m)) == b)
                break;
        }

        if (m == l)
            goto L220;

L130:
        if (j == 30)
            goto L1000;
        j++;
        l1 = l + 1;
        l2 = l1 + 1;
        g = D(l);
        p = (D(l1) - g) / (2.0f * E(l));
        r = pythag_s(p, 1.0f);
        D(l) = E(l) / (p + copysignf(r, p));
        D(l1) = E(l) * (p + copysignf(r, p));
        dl1 = D(l1);
        h = g - D(l);
        if (l2 <= N) {
            for (i = l2; i <= N; i++)
                D(i) -= h;
        }

        f += h;
        p = D(m);
        c = 1.0f;
        c2 = c;
        el1 = E(l1);
        s = 0.0f;
        mml = m - l;

        for (ii = 1; ii <= mml; ii++) {
            c3 = c2;
            c2 = c;
            s2 = s;
            i = m - ii;
            g = c * E(i);
            h = c * p;
            if (fabsf(p) < fabsf(E(i))) {
                c = p / E(i);
                r = sqrtf(c * c + 1.0f);
                E(i + 1) = s * E(i) * r;
                s = 1.0f / r;
                c = c * s;
            } else {
                c = E(i) / p;
                r = sqrtf(c * c + 1.0f);
                E(i + 1) = s * p * r;
                s = c / r;
                c = 1.0f / r;
            }
            p = c * D(i) - s * g;
            D(i + 1) = h + s * (c * g + s * D(i));

            for (k = 1; k <= N; k++) {
                h = Z(k, i + 1);
                Z(k, i + 1) = s * Z(k, i) + c * h;
                Z(k, i) = c * Z(k, i) - s * h;
            }
        }

        p = -s * s2 * c3 * el1 * E(l) / dl1;
        E(l) = s * p;
        D(l) = c * p;
        if (b + fabsf(E(l)) > b)
            goto L130;

L220:
        D(l) += f;
    }

    for (ii = 2; ii <= N; ii++) {
        i = ii - 1;
        k = i;
        p = D(i);
        for (j = ii; j <= N; j++) {
            if (D(j) < p) {
                k = j;
                p = D(j);
            }
        }
        if (k == i)
            continue;
        D(k) = D(i);
        D(i) = p;
        for (j = 1; j <= N; j++) {
            p = Z(j, i);
            Z(j, i) = Z(j, k);
            Z(j, k) = p;
        }
    }

    goto L1001;
L1000:
    *ierr = l;
L1001:

#undef Z
#undef D
#undef E
    return;
}

/* Eigenvalues of a symmetric tridiagonal matrix by the rational QL method
 * (eigenvalues-only path; unreachable from pc.f/project.f, which both pass
 * matz=1 and so always take the tred2_s/tql2_s path -- kept for ABI
 * completeness but not exercised by the test suite). */
static void tqlrat_s(int *n, float *d, float *e2, int *ierr)
{
    int N = *n;
    int i, j, l, m, ii, l1, mml;
    float b, c, f, g, h, p, r, s;
    const float machep = FLT_EPSILON;

#define D(i) d[(i) - 1]
#define E2(i) e2[(i) - 1]

    *ierr = 0;
    if (N == 1)
        goto L1001;

    for (i = 2; i <= N; i++)
        E2(i - 1) = E2(i);

    f = 0.0f;
    b = 0.0f;
    c = 0.0f;
    E2(N) = 0.0f;

    for (l = 1; l <= N; l++) {
        j = 0;
        h = machep * (fabsf(D(l)) + sqrtf(E2(l)));
        if (b <= h) {
            b = h;
            c = b * b;
        }

        for (m = l; m <= N; m++) {
            if (E2(m) <= c)
                break;
        }

        if (m == l)
            goto L210;

L130:
        if (j == 30)
            goto L1000;
        j++;
        l1 = l + 1;
        s = sqrtf(E2(l));
        g = D(l);
        p = (D(l1) - g) / (2.0f * s);
        r = pythag_s(p, 1.0f);
        D(l) = s / (p + copysignf(r, p));
        h = g - D(l);

        for (i = l1; i <= N; i++)
            D(i) -= h;

        f += h;
        g = D(m);
        if (g == 0.0f)
            g = b;
        h = g;
        s = 0.0f;
        mml = m - l;

        for (ii = 1; ii <= mml; ii++) {
            i = m - ii;
            p = g * h;
            r = p + E2(i);
            E2(i + 1) = s * r;
            s = E2(i) / r;
            D(i + 1) = h + s * (h + D(i));
            g = D(i) - E2(i) / g;
            if (g == 0.0f)
                g = b;
            h = g * p / r;
        }

        E2(l) = s * g;
        D(l) = h;
        if (h == 0.0f)
            goto L210;
        if (fabsf(E2(l)) <= fabsf(c / h))
            goto L210;
        E2(l) = h * E2(l);
        if (E2(l) != 0.0f)
            goto L130;

L210:
        p = D(l) + f;
        if (l == 1) {
            i = 1;
        } else {
            i = 0;
            for (ii = 2; ii <= l; ii++) {
                i = l + 2 - ii;
                if (p >= D(i - 1))
                    break;
                D(i) = D(i - 1);
                i = 0;
            }
            if (i == 0)
                i = 1;
        }
        D(i) = p;
    }

    goto L1001;
L1000:
    *ierr = l;
L1001:

#undef D
#undef E2
    return;
}

/* Eigenvalues and, optionally, eigenvectors of a real symmetric matrix.
 * Direct EISPACK dispatcher: matz==0 -> tred1_s/tqlrat_s (eigenvalues only),
 * matz!=0 -> tred2_s/tql2_s (eigenvalues and eigenvectors). fv1 and fv2 are
 * caller-supplied scratch of length n; nothing here is allocated. */
void rs_(int *nm, int *n, float *a, float *w, int *matz, float *z,
          float *fv1, float *fv2, int *ierr)
{
    if (*n > *nm) {
        *ierr = 10 * (*n);
        return;
    }

    if (*matz == 0) {
        tred1_s(nm, n, a, w, fv1, fv2);
        tqlrat_s(n, w, fv2, ierr);
        return;
    }

    tred2_s(nm, n, a, w, fv1, z);
    tql2_s(nm, n, w, fv1, z, ierr);
}
