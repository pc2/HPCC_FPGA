#include "parameters.h"

/**
 * Simple reference kernel for matrix transposition.
 * Not optimized and only using global memory.
 *
 * Will do the following:
 *
 * A_out = trans(A) + B
 *
 * where A_out, A and B are matrices of size matrixSize*matrixSize
 *
 * @param A Buffer for matrix A
 * @param B Buffer for matrix B
 * @param A_out Output buffer for result matrix
 * @param matrixSize Size of the matrices
 */
__kernel
void transpose(__global DEVICE_DATA_TYPE *restrict A,
               __global DEVICE_DATA_TYPE *restrict B,
               __global DEVICE_DATA_TYPE *restrict A_out,
               uint matrixSize) {
    for (int i = 0; i < matrixSize; i++) {
        for (int j = 0; j < matrixSize; j++) {
            A_out[i * matrixSize + j] = A[j * matrixSize + i] + B[i * matrixSize + j];
        }
    }
}