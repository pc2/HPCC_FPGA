/*
Copyright (c) 2019 Marius Meyer

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#include "parameters.h"

#define BLOCK_SIZE (1 << LOCAL_MEM_BLOCK_LOG)
#define GEMM_BLOCK (1 << REGISTER_BLOCK_LOG)

__attribute__((uses_global_work_offset(0)))
__kernel
void
gefa(__global DEVICE_DATA_TYPE* restrict a,
	unsigned n_blocks) {

	const int n = n_blocks * BLOCK_SIZE;

	for (int k = 0; k < n-1; k++) {

        DEVICE_DATA_TYPE scale_a = a[k*n+k];
        DEVICE_DATA_TYPE inv_scale_a = -1.0 / scale_a;

        // For each element below it
        for (int i=0; i < n; i++) {
            a[k * n + i] = (i > k) ? a[k * n + i] * inv_scale_a : ((i == k) ? scale_a : a[k * n + i]);
        }
        // For each column right of current diagonal element
        for (int j = k + 1; j < n; j++) {
            // For each element below it
            for (int i = k+1; i < n; i++) {
                a[j * n + i] += a[k * n + i] * a[j * n + k];
            }
        }
 	}
}


__attribute__((uses_global_work_offset(0)))
__kernel
void
gesl(__global DEVICE_DATA_TYPE* restrict a, 
	__global DEVICE_DATA_TYPE* restrict b,
	unsigned n_blocks) {

    const int n = n_blocks * BLOCK_SIZE;

	// solve l*y = b
	// For each row in matrix
	for (int k = 0; k < n - 1; k++) {
		// For each row below add
		for (int i = k + 1; i < n; i++) {
			// add solved upper row to current row
			b[i] += b[k] * a[n * k + i];
		}
	}

	// now solve  u*x = y
	for (int k = n - 1; k >= 0; k--) {
		b[k] = b[k] / a[n * k + k];
		for (int i = 0; i < k; i++) {
			b[i] -= b[k] * a[n * k + i];
		}
	}
}