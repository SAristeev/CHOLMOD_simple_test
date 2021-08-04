#pragma warning(disable : 4996)
#include "cholmod_function.h"
#include "cholmod_internal.h"
#include "cholmod_check.h"
#include "cholmod.h"

#define USE_GPU 1

#if USE_GPU
#define CHOLMOD(x) cholmod_l_##x
#else
#define CHOLMOD(x) cholmod_##x
#endif

int main(void)
{
    FILE* in;
    in = fopen("../input/A.tri", "r");
    if (in == NULL) return -1;
    cholmod_sparse* A;
    cholmod_dense* x, * b, * r;
    cholmod_factor* L;
    double one[2] = { 1,0 }, m1[2] = { -1,0 };	    /* basic scalars */
    cholmod_common* c = (cholmod_common*)malloc(sizeof(cholmod_common));
    cholmod_start(c);			    /* start CHOLMOD */
    c->useGPU = 1;
    c->supernodal = CHOLMOD_SUPERNODAL;
    A = cholmod_read_sparse(in, c);	    /* read in a matrix */
    fclose(in);
    cholmod_print_sparse(A, "A", c);		    /* print the matrix */
    if (A == NULL || A->stype == 0)		    /* A must be symmetric */
    {
        cholmod_free_sparse(&A, c);
        cholmod_finish(c);
        return (0);
    }
    b = cholmod_ones(A->nrow, 1, A->xtype, c);   /* b = ones(n,1) */
    L = cholmod_analyze(A, c);		    /* analyze */
    cholmod_factorize(A, L, c);		    /* factorize */
    x = cholmod_solve(CHOLMOD_A, L, b, c);	    /* solve Ax=b */
    r = cholmod_copy_dense(b, c);		    /* r = b */
#ifndef NMATRIXOPS
    cholmod_sdmult(A, 0, m1, one, x, r, c);	    /* r = r-Ax */
    printf("norm(b-Ax) %8.1e\n",
        cholmod_norm_dense(r, 0, c));	    /* print norm(r) */
#else
    printf("residual norm not computed (requires CHOLMOD/MatrixOps)\n");
#endif
    cholmod_free_factor(&L, c);		    /* free matrices */
    cholmod_free_sparse(&A, c);
    cholmod_free_dense(&r, c);
    cholmod_free_dense(&x, c);
    cholmod_free_dense(&b, &c);
    cholmod_finish(c);			    /* finish CHOLMOD */
    return (0);
}
