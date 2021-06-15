#ifndef GMRES_H
#define GMRES_H


/**
 * @brief Reference GMRES implementation taken from https://bitbucket.org/icl/hpl-ai/src/master/gmres.c
 * 
 * @param n heigth of the matrix A
 * @param A pointer to matrix A
 * @param lda width of matrix A
 * @param x 
 * @param b 
 * @param LU 
 * @param ldlu 
 * @param restart 
 * @param max_it 
 * @param tol 
 */
void gmres_ref(int n, double* A, int lda, double* x, double* b, double* LU, int ldlu, int restart, int max_it, double tol);


#endif