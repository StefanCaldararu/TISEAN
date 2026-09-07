#include <math.h>
#include <stdlib.h>
#include <stdio.h>

/* FFTPACK Ogg version c implementation (defined in /source_c/routines/fft.c)*/
void __ogg_fdrffti(int n, float *wsave, int *ifac);
void __ogg_fdrfftf(int n, float *r, float *wsave, int *ifac);
void __ogg_fdrfftb(int n, float *r, float *wsave, int *ifac);

#define TOSPEC_NX 1000000

float tospec_(int *nmax, float *a, float *x, int *ibin)
{
    int N = *nmax;
    int bin = *ibin;
    int n, i, ib;

    if (N > TOSPEC_NX) {
        fprintf(stderr, "tospec: make nx larger.\n");
        exit(1);
    }

    float *w = (float*)malloc(sizeof(float) * N);
    float *wsave = (float*)malloc(sizeof(float) * (2*N + 10));
    int ifac[15];

    if (!w || !wsave) {
        fprintf(stderr, "tospec: malloc failed\n");
        exit(1);
    }

    for (n = 1; n <= N; n++)
        w[n-1] = x[n-1];

    __ogg_fdrffti(N, wsave, ifac);
    __ogg_fdrfftf(N, x, wsave, ifac);

    for (n = 1; n <= N; n++)
        x[n-1] = x[n-1] / (float)N;

    x[0] = x[0] * (a[0] / (x[0]*x[0]));

    for (i = 2+bin; i <= (N+1)/2 - bin; i += 2*bin+1) {
        float p = 0.0f;
        float ab;

        for (ib = i-bin; ib <= i+bin; ib++)
            p = p + x[2*ib-3]*x[2*ib-3] + x[2*ib-2]*x[2*ib-2];

        ab = a[2*i-3] / p;

        for (ib = i-bin; ib <= i+bin; ib++) {
            x[2*ib-3] = x[2*ib-3] * ab;
            x[2*ib-2] = x[2*ib-2] * ab;
        }
    }

    if (N % 2 == 0)
        x[N-1] = x[N-1] * (a[N-1] / (x[N-1]*x[N-1]));

    __ogg_fdrfftb(N, x, wsave, ifac);

    float result = 0.0f;
    for (n = 1; n <= N; n++)
        result = result + (x[n-1]-w[n-1])*(x[n-1]-w[n-1]);
    result = sqrtf(result / (float)N);

    free(w);
    free(wsave);

    return result;
}
