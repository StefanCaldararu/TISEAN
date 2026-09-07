/* dqk15.c
 *
 * C replacement for SLATEC/QUADPACK dqk15.f
 *
 * Provides:
 *   dqk15_
 *
 * ABI-compatible with gfortran-generated symbols. The integrand F is
 * itself a Fortran function, passed by gfortran as a plain code
 * pointer taking its argument by reference.
 */

#include <float.h>
#include <math.h>

/* GAUSS QUADRATURE WEIGHTS AND KRONROD QUADRATURE ABSCISSAE AND WEIGHTS
 * AS EVALUATED WITH 80 DECIMAL DIGIT ARITHMETIC BY L. W. FULLERTON,
 * BELL LABS, NOV. 1981.
 */

static const double wg[4] = {
    0.129484966168869693270611432679082,
    0.279705391489276667901467771423780,
    0.381830050505118944950369775488975,
    0.417959183673469387755102040816327
};

static const double xgk[8] = {
    0.991455371120812639206854697526329,
    0.949107912342758524526189684047851,
    0.864864423359769072789712788640926,
    0.741531185599394439863864773280788,
    0.586087235467691130294144838258730,
    0.405845151377397166906606412076961,
    0.207784955007898467600689403773245,
    0.000000000000000000000000000000000
};

static const double wgk[8] = {
    0.022935322010529224963732008058970,
    0.063092092629978553290700663189204,
    0.104790010322250183839876322541518,
    0.140653259715525918745189590510238,
    0.169004726639267902826583426598550,
    0.190350578064785409913256402421014,
    0.204432940075298892414161999234649,
    0.209482141084727828012999174891714
};

void dqk15_(double (*f)(double *), double *a, double *b, double *result,
            double *abserr, double *resabs, double *resasc)
{
    double epmach = DBL_EPSILON;
    double uflow = DBL_MIN;

    double centr = 0.5e+00 * (*a + *b);
    double hlgth = 0.5e+00 * (*b - *a);
    double dhlgth = fabs(hlgth);

    double fv1[7], fv2[7];
    double fc, resg, resk, reskh;
    int j, jtw, jtwm1;

    /*           COMPUTE THE 15-POINT KRONROD APPROXIMATION TO
     *           THE INTEGRAL, AND ESTIMATE THE ABSOLUTE ERROR.
     */

    fc = f(&centr);
    resg = fc * wg[3];
    resk = fc * wgk[7];
    *resabs = fabs(resk);
    for (j = 1; j <= 3; j++) {
        double absc, fval1, fval2, fsum;
        jtw = j * 2;
        absc = hlgth * xgk[jtw - 1];
        {
            double xm = centr - absc;
            double xp = centr + absc;
            fval1 = f(&xm);
            fval2 = f(&xp);
        }
        fv1[jtw - 1] = fval1;
        fv2[jtw - 1] = fval2;
        fsum = fval1 + fval2;
        resg = resg + wg[j - 1] * fsum;
        resk = resk + wgk[jtw - 1] * fsum;
        *resabs = *resabs + wgk[jtw - 1] * (fabs(fval1) + fabs(fval2));
    }
    for (j = 1; j <= 4; j++) {
        double absc, fval1, fval2, fsum;
        jtwm1 = j * 2 - 1;
        absc = hlgth * xgk[jtwm1 - 1];
        {
            double xm = centr - absc;
            double xp = centr + absc;
            fval1 = f(&xm);
            fval2 = f(&xp);
        }
        fv1[jtwm1 - 1] = fval1;
        fv2[jtwm1 - 1] = fval2;
        fsum = fval1 + fval2;
        resk = resk + wgk[jtwm1 - 1] * fsum;
        *resabs = *resabs + wgk[jtwm1 - 1] * (fabs(fval1) + fabs(fval2));
    }
    reskh = resk * 0.5e+00;
    *resasc = wgk[7] * fabs(fc - reskh);
    for (j = 1; j <= 7; j++) {
        *resasc = *resasc + wgk[j - 1] * (fabs(fv1[j - 1] - reskh) + fabs(fv2[j - 1] - reskh));
    }
    *result = resk * hlgth;
    *resabs = *resabs * dhlgth;
    *resasc = *resasc * dhlgth;
    *abserr = fabs((resk - resg) * hlgth);
    if (*resasc != 0.0e+00 && *abserr != 0.0e+00)
        *abserr = *resasc * fmin(1.0e+00, pow(2.0e+02 * *abserr / *resasc, 1.5e+00));
    if (*resabs > uflow / (0.5e+02 * epmach))
        *abserr = fmax((epmach * 0.5e+02) * *resabs, *abserr);
}
