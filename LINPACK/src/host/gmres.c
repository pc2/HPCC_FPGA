// Innovative Computing Laboratory
// University of Tennessee
// Copyright (c) 2019-2020
  
// All rights reserved.

// Redistribution and use in source and binary forms, with or without modification, are permitted (subject to the limitations in the disclaimer below) provided that the following conditions are met:

//         * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
//         * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
//         * Neither the name of University of Tennessee nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

// NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>

#define HPLAI_INDEX2D(PTR, R, C, LDIM) ( ((PTR) + (R)) + sizeof(char) * (C) / sizeof(char) * (LDIM) )
#define A(i, j) *HPLAI_INDEX2D(A, (i), (j), lda)
#define H(i, j) *HPLAI_INDEX2D(H, (i), (j), (m + 1))
#define V(i, j) *HPLAI_INDEX2D(V, (i), (j), n)

extern void dtrsm_ref(char side, char uplo, char transa, char diag, int m, int n,
           double alpha, double* A, int lda, double* B, int ldb);

extern void dgemv_ref(char trans, int m, int n, double alpha, double* A,
           int lda, double* X, int incx, double beta, double* Y,
           int incy);

extern double dlange_ref(char norm, int m, int n, double* A, int lda);

// Compute Gevens rotation matrix parameters.
void 
rotmat(double a, double b, double* c, double* s) {
    if (b == 0.0) {
        *c = 1.0;
        *s = 0.0;
    } else if (fabs(b) > fabs(a)) {
        double temp = a / b;
        *s = 1.0 / sqrt(1.0 + temp * temp);
        *c = temp * *s;
    } else {
        double temp = b / a;
        *c = 1.0 / sqrt(1.0 + temp * temp);
        *s = temp * *c;
    }
}

// Flexible generalized minimal residual method (FGMRES)
// Based on http://www.netlib.org/templates/matlab/gmres.m
void 
gmres_ref(int n, double* A, int lda, double* x, double* b, double* LU,
           int ldlu, int restart, int max_it, double tol) {

    int i, j, k, iter;
    int updated;  // Flag to show if x is updated or not.

    int m = restart;
    if (m > n) {
        m = n;
    }
    double temp;

    double* cs = (double*)malloc(m * sizeof(double));
    double* e1 = (double*)malloc(n * sizeof(double));
    double* r = (double*)malloc(n * sizeof(double));
    double* s = (double*)malloc((m + 1) * sizeof(double));
    double* sn = (double*)malloc(m * sizeof(double));
    double* w = (double*)malloc(n * sizeof(double));
    double* old_x = (double*)malloc(n * sizeof(double));

    double* H = (double*)malloc(m * (m + 1) * sizeof(double));
    double* V = (double*)malloc(n * (m + 1) * sizeof(double));

    memset(cs, 0, m * sizeof(double));
    memset(e1, 0, n * sizeof(double));
    e1[0] = 1.0;
    memset(r, 0, n * sizeof(double));
    memset(s, 0, (m + 1) * sizeof(double));
    memset(sn, 0, m * sizeof(double));

    memset(H, 0, m * (m + 1) * sizeof(double));
    memset(V, 0, n * (m + 1) * sizeof(double));

    memcpy(old_x, x, n * sizeof(double));


    double norm_b = dlange_ref('F', n, 1, b, n);
    if (norm_b == 0.0) {
        norm_b = 1.0;
    }

    printf("%f\n", norm_b);

    // r = U \ (L \ (b - A*x))
    memcpy(r, b, n * sizeof(double));
    dgemv_ref('N', n, n, -1.0, A, lda, x, 1, 1.0, r, 1);
    dtrsm_ref('L', 'L', 'N', 'U', n, 1, 1.0, LU, ldlu, r, n);
    dtrsm_ref('L', 'U', 'N', 'N', n, 1, 1.0, LU, ldlu, r, n);

    // TODO Residual seems already to be wrong here!
    double error = dlange_ref('F', n, 1, r, n) / norm_b;
    printf("Residual norm at the beginning of GMRES: %e\n", error);

    if (error < tol) {
        return;
    }

    // Begin iteration
    for (iter = 0; iter < max_it; ++iter) {

        updated = 0;

        // r = U \ (L \ (b - A*x))
        if (iter != 0) {
            memcpy(r, b, n * sizeof(double));
            dgemv_ref('N', n, n, -1.0, A, lda, x, 1, 1.0, r, 1);
            dtrsm_ref('L', 'L', 'N', 'U', n, 1, 1.0, LU, ldlu, r, n);
            dtrsm_ref('L', 'U', 'N', 'N', n, 1, 1.0, LU, ldlu, r, n);
        }

        // V0 = r; s[0] = |r|;
        double norm_r = dlange_ref('F', n, 1, r, n);
        for (i = 0; i < n; i++) {
            r[i] /= norm_r;
        }
        memcpy(V, r, n * sizeof(double));
        s[0] = norm_r;

        for (i = 0; i < m; i++) {
            // w = U \  (L \ (A * Vi))
            dgemv_ref('N', n, n, 1.0, A, lda, V + i * n, 1, 0.0, w, 1);
            dtrsm_ref('L', 'L', 'N', 'U', n, 1, 1.0, LU, ldlu, w, n);
            dtrsm_ref('L', 'U', 'N', 'N', n, 1, 1.0, LU, ldlu, w, n);

            // Gram-Schmidt process
            for (k = 0; k <= i; k++) {
                // H(k,i) = w' * V(:,i)
                for (j = 0; j < n; j++) {
                    H(k,i) += w[j] * V(j, k);
                }
                // w = w - H(k,i) * V(:,i)
                for (j = 0; j < n; j++) {
                    w[j] -= H(k, i) * V(j, k);
                }
            }
            H(i + 1, i) = dlange_ref('F', n, 1, w, n);

            for (j = 0; j < n; j++) {
                w[j] /= H(i + 1, i);
                V(j, i + 1) = w[j];
            }

            // Apply givens rotation
            for (k = 0; k < i; k++) {
                temp = cs[k] * H(k, i) + sn[k] * H(k + 1, i);
                H(k + 1, i) = -sn[k] * H(k, i) + cs[k] * H(k + 1, i);
                H(k, i) = temp;
            }

            // Find i-th rotation
            rotmat(H(i, i), H(i + 1, i), cs + i, sn + i);

            // Approximate residual norm
            temp = cs[i] * s[i];
            s[i + 1] = -sn[i] * s[i];
            s[i] = temp;
            H(i, i) = cs[i] * H(i, i) + sn[i] * H(i + 1, i);
            H(i + 1, i) = 0.0;

            error = fabs(s[i + 1]) / norm_b;
            printf(
                "Estimated residual norm at the %d-th iteration of GMRES: "
                "%e\n",
                i + 1, error);
            if (error <= tol) {
                memcpy(w, s, (i + 1) * sizeof(double));
                dtrsm_ref('L', 'U', 'N', 'N', i + 1, 1, 1.0, H, m + 1, w, n);
                dgemv_ref('N', n, i + 1, 1.0, V, n, w, 1, 1.0, x, 1);
                updated = 1;

                // Check the HPL-AI scaled residual
                double norm_A = dlange_ref('I', n, n, A, lda);
                double norm_x = dlange_ref('I', n, 1, x, n);
                double norm_b = dlange_ref('I', n, 1, b, n);
                memcpy(r, b, n * sizeof(double));
                dgemv_ref('N', n, n, 1.0, A, lda, x, 1, -1.0, r, 1);
                double threshold = 16.0;
                double eps = DBL_EPSILON / 2;
                double error = dlange_ref('I', n, 1, r, n) /
                               (norm_A * norm_x + norm_b) / n / eps;

                // Continue GMRES if it didn't pass the threshold
                if (error > threshold) {
                    memcpy(x, old_x, n * sizeof(double));
                    updated = 0;
                    continue;
                }
                break;
            }
        }

        // Update approximation
        // w = H \ s
        // x = V * w + x
        if (!updated) {
            memcpy(w, s, m * sizeof(double));
            dtrsm_ref('L', 'U', 'N', 'N', m, 1, 1.0, H, m + 1, w, n);
            dgemv_ref('N', n, m, 1.0, V, n, w, 1, 1.0, x, 1);
        }

        // Compute redisual
        memcpy(r, b, n * sizeof(double));
        dgemv_ref('N', n, n, -1.0, A, lda, x, 1, 1.0, r, 1);
        dtrsm_ref('L', 'L', 'N', 'U', n, 1, 1.0, LU, ldlu, r, n);
        dtrsm_ref('L', 'U', 'N', 'N', n, 1, 1.0, LU, ldlu, r, n);
        norm_r = dlange_ref('F', n, 1, r, n);
        error = norm_r / norm_b;
        if (error <= tol) {
            break;
        }
    }

    free(cs);
    free(e1);
    free(r);
    free(s);
    free(sn);
    free(w);

    free(H);
    free(V);
}
