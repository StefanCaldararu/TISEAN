#include <math.h>
#include <stdlib.h>
#include <stdio.h>

/* FFTPACK Ogg version c implementation (defined in /source_c/routines/fft.c)*/
void __ogg_fdrffti(int n, float *wsave, int *ifac);
void __ogg_fdrfftf(int n, float *r, float *wsave, int *ifac);
void __ogg_fdrfftb(int n, float *r, float *wsave, int *ifac);

void store_spec_(int *nmax, float *x, int *iback)
{
    int N = *nmax;
    int back = *iback;
    if(N <= 1) return;

    float *wsave = (float*)malloc(sizeof(float) * (2*N + 10));
    int ifac[15];

    if(!wsave) {
        fprintf(stderr, "store_spec: malloc failed\n");
        exit(1);
    }

    __ogg_fdrffti(N, wsave, ifac);
    __ogg_fdrfftf(N, x, wsave, ifac);

    for(int i = 0; i < N; i++)
        x[i] /= (float)N;

    x[0] = x[0] * x[0];

    int n2 = (N + 1) / 2;

    for(int k = 2; k <= n2; k++)
    {
        int re_i = 2*k - 3;
        int im_i = 2*k - 2;

        float re = x[re_i];
        float im = x[im_i];

        float amp = re*re + im*im;
        float pha = atan2f(im, re);

        x[re_i] = amp;
        x[im_i] = pha;
    }

    if(N % 2 == 0)
    {
        x[N - 1] = x[N - 1] * x[N - 1];
    }

    if(back != 0)
    {
        for(int i = 0; i < N; i++)
            x[i] *= N;

        for(int k = 2; k <= n2; k++)
            x[2*k - 1] = 0.0f;

        __ogg_fdrfftb(N, x, wsave, ifac);
    }

    free(wsave);
}