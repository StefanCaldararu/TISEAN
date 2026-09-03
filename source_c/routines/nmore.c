/* nmore.c
 *
 * C replacement for TISEAN nmore.f
 *
 * Provides:
 *   nmore_
 *   nless_
 *
 * ABI-compatible with gfortran-generated symbols.
 */

/*------------------------------------------------------------------*/
/* isfact                                                            */
/*                                                                   */
/* determine if n is factorisable using the first nprimes primes     */
/*------------------------------------------------------------------*/

static int isfact(int n)
{
    static const int iprime[] = {2, 3, 5};
    static const int nprimes = 3;
    int ncur = n;
    int i;

    while (ncur != 1) {
        int divided = 0;

        for (i = 0; i < nprimes; i++) {
            if (ncur % iprime[i] == 0) {
                ncur /= iprime[i];
                divided = 1;
                break;
            }
        }

        if (!divided)
            return 0;
    }

    return 1;
}

/*------------------------------------------------------------------*/
/* nmore_                                                            */
/*                                                                   */
/* find smallest factorisable number .ge.n                           */
/*------------------------------------------------------------------*/

int nmore_(int *n)
{
    int result = *n;

    while (!isfact(result))
        result++;

    return result;
}

/*------------------------------------------------------------------*/
/* nless_                                                            */
/*                                                                   */
/* find largest factorisable number .le.n                            */
/*------------------------------------------------------------------*/

int nless_(int *n)
{
    int result = *n;

    while (!isfact(result))
        result--;

    return result;
}
