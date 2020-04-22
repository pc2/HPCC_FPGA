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

/**
Store a block to global memory

@param a_block local memory buffer to load the block from
@param a the global memory buffer of the Matrix
@param x_block x position of the block
@param y_block y position of the block
@param lda_block LDA of the matrix in number of blocks
*/
void
store_block(DEVICE_DATA_TYPE a_block[BLOCK_SIZE][BLOCK_SIZE],
            global DEVICE_DATA_TYPE* restrict a,
            uint x_block, uint y_block, uint lda_block) {

    for (int i = 0; i < BLOCK_SIZE; i++) {
#pragma unroll GLOBAL_MEM_UNROLL
        for (int j = 0; j < BLOCK_SIZE; j++) {
            a[(y_block * lda_block * BLOCK_SIZE + x_block) * BLOCK_SIZE + j
              + i * lda_block * BLOCK_SIZE] = a_block[i][j];
        }
    }
}


/**
Calculate for the Level 2 block:

c = c +  a * b

where a,b,c are matrices of size GEMM_BLOCK.
Calculation itself is fully unrolled.
 */
void register_gemm(const DEVICE_DATA_TYPE a[GEMM_BLOCK][GEMM_BLOCK],
                    const DEVICE_DATA_TYPE b[GEMM_BLOCK][GEMM_BLOCK],
                    DEVICE_DATA_TYPE c_out[GEMM_BLOCK][GEMM_BLOCK]) {

    DEVICE_DATA_TYPE a_block[GEMM_BLOCK][GEMM_BLOCK + 1];
    DEVICE_DATA_TYPE b_block[GEMM_BLOCK + 1][GEMM_BLOCK];
    DEVICE_DATA_TYPE c_block[GEMM_BLOCK][GEMM_BLOCK];

    // Load block of matrix A and B and init C and reorder values
#pragma unroll
    for (int y=0; y<GEMM_BLOCK; y++) {
#pragma unroll
        for (int x=0; x<GEMM_BLOCK; x++) {
            int k = (x + y) % GEMM_BLOCK;
            a_block[y][x] = a[y][k];
            b_block[y][x] = b[k][x];
            c_block[y][x] = 0;
        }
    }

    // Calculate result for 8x8 matrix
#pragma unroll
    for (int i=0;i<GEMM_BLOCK; i++) {
#pragma unroll
        for (int x=0; x<GEMM_BLOCK;x++) {
            a_block[x][GEMM_BLOCK] = a_block[x][0];
            b_block[GEMM_BLOCK][x] = b_block[0][x];
        }
#pragma unroll
        for(int y=0; y < GEMM_BLOCK; y++) {
#pragma unroll
            for (int x=0; x<GEMM_BLOCK;x++) {
                c_block[y][x] += a_block[y][x] * b_block[y][x];
                a_block[y][x] = a_block[y][x + 1];
                b_block[y][x] = b_block[y + 1][x];
            }
        }
    }

#pragma unroll
    for(int y=0; y < GEMM_BLOCK; y++) {
#pragma unroll
        for (int x=0; x<GEMM_BLOCK;x++) {
            c_out[y][x] += c_block[y][x];
        }
    }
}


/**
GEMM for the Level 1 Block


@param left_block Most left block that was modified by C2 before
@param top_block Most upper block that was modified by C3 before
@param current_block_in Current input block
@param current_block_out Block to write the output to
*/
void
local_gemm(const DEVICE_DATA_TYPE a_block[BLOCK_SIZE / GEMM_BLOCK][BLOCK_SIZE / GEMM_BLOCK]
                                        [GEMM_BLOCK][GEMM_BLOCK],
           const DEVICE_DATA_TYPE b_block[BLOCK_SIZE / GEMM_BLOCK][BLOCK_SIZE / GEMM_BLOCK]
                                            [GEMM_BLOCK][GEMM_BLOCK],
           DEVICE_DATA_TYPE c_block_out[BLOCK_SIZE / GEMM_BLOCK][BLOCK_SIZE / GEMM_BLOCK]
                                        [GEMM_BLOCK][GEMM_BLOCK],
           DEVICE_DATA_TYPE alpha,
           const bool first_block) {

#pragma loop_coalesce 2
    // For each column in top block
    for (int i = 0; i < BLOCK_SIZE / GEMM_BLOCK; i++) {
        // For each element below it in current block
        for (int j = 0; j < BLOCK_SIZE / GEMM_BLOCK; j++) {
            DEVICE_DATA_TYPE   tmp_small_block_out[GEMM_BLOCK][GEMM_BLOCK];
#pragma unroll
            for (int ii = 0; ii < GEMM_BLOCK; ii++) {
#pragma unroll
                for (int jj = 0; jj < GEMM_BLOCK; jj++) {
                    tmp_small_block_out[ii][jj] = 0;
                }
            }
            // For each diagonal element in left block
            for (int k=0; k < BLOCK_SIZE / GEMM_BLOCK; k++) {
                register_gemm(a_block[i][k], b_block[k][j],
                               tmp_small_block_out);
            }

#pragma unroll
            for (int ii = 0; ii < GEMM_BLOCK; ii++) {
#pragma unroll
                for (int jj = 0; jj < GEMM_BLOCK; jj++) {
                    c_block_out[i][j][ii][jj] = first_block ? c_block_out[i][j][ii][jj] + alpha * tmp_small_block_out[ii][jj] : alpha * tmp_small_block_out[ii][jj];
                }
            }
        }
    }
}


/**
Two level blocked GEMM kernel

calculates C_OUT = alpha * C + beta * A.dot(B)

@param a The data array representing the whole matrix a in global memory
@param b The data array representing the whole matrix b in global memory
@param c The data array representing the whole matrix c in global memory
@param c_out The data array that will used as output of the result
@param alpha The alpha scalar value
@param beta The beta scalar value
@param a_size the x and y size of the matrix in blocks
*/
__attribute__((uses_global_work_offset(0)))
__kernel
void gemm(__global volatile const DEVICE_DATA_TYPE* restrict a,
          __global volatile const DEVICE_DATA_TYPE* restrict b,
          __global volatile const DEVICE_DATA_TYPE* restrict c,
          __global volatile DEVICE_DATA_TYPE* restrict c_out,
          const DEVICE_DATA_TYPE alpha,
          const DEVICE_DATA_TYPE beta,
          const uint size) {

    const unsigned a_size = size / BLOCK_SIZE;

    // Level 1 Matrix Multiplication
#pragma loop_coalesce 2
#pragma disable_loop_pipelining
    for (int x_block = 0; x_block < a_size; x_block++) {
#pragma disable_loop_pipelining
        for (int y_block = 0; y_block < a_size; y_block++) {
            DEVICE_DATA_TYPE c_block[BLOCK_SIZE / GEMM_BLOCK][BLOCK_SIZE / GEMM_BLOCK]
            [GEMM_BLOCK][GEMM_BLOCK]  __attribute((numbanks(GEMM_BLOCK * GEMM_BLOCK)));

            for (int diagonal_block=0; diagonal_block < a_size; diagonal_block++) {
                DEVICE_DATA_TYPE a_block[BLOCK_SIZE / GEMM_BLOCK][BLOCK_SIZE / GEMM_BLOCK]
                                        [GEMM_BLOCK][GEMM_BLOCK]  __attribute((numbanks(GEMM_BLOCK * GEMM_BLOCK)));
                DEVICE_DATA_TYPE b_block[BLOCK_SIZE / GEMM_BLOCK][BLOCK_SIZE / GEMM_BLOCK]
                                        [GEMM_BLOCK][GEMM_BLOCK]  __attribute((numbanks(GEMM_BLOCK * GEMM_BLOCK)));
                // Load all needed level 1 blocks
#pragma loop_coalesce 2
                for (int i = 0; i < BLOCK_SIZE ; i++) {
                    for (int j = 0; j < BLOCK_SIZE / GLOBAL_MEM_UNROLL; j++) {
                        DEVICE_DATA_TYPE a_reorder_buffer[GLOBAL_MEM_UNROLL];
                        DEVICE_DATA_TYPE b_reorder_buffer[GLOBAL_MEM_UNROLL];
#pragma unroll
                        for (int u = 0; u < GLOBAL_MEM_UNROLL; u++) {
                            a_reorder_buffer[u] = a[(y_block * size + diagonal_block) * BLOCK_SIZE +
                                j * GLOBAL_MEM_UNROLL + u + i * size];
                            b_reorder_buffer[u] = b[(diagonal_block * size + x_block) * BLOCK_SIZE +
                                                          j * GLOBAL_MEM_UNROLL + u + i * size];
                        }
#pragma unroll
                        for (int b = 0; b < GLOBAL_MEM_UNROLL/GEMM_BLOCK; b++) {
#pragma unroll
                            for (int u = 0; u < GEMM_BLOCK; u++) {
                                a_block[i / GEMM_BLOCK][j * (GLOBAL_MEM_UNROLL / GEMM_BLOCK)+ b][i & (GEMM_BLOCK - 1)][u] = a_reorder_buffer[b * GEMM_BLOCK + u];
                                b_block[i / GEMM_BLOCK][j * (GLOBAL_MEM_UNROLL / GEMM_BLOCK)+ b][i & (GEMM_BLOCK - 1)][u] = b_reorder_buffer[b * GEMM_BLOCK + u];
                            }
                        }
                    }
                }
                local_gemm(a_block, b_block, c_block, alpha, diagonal_block);
            }

#pragma loop_coalesce
            for (int i = 0; i < BLOCK_SIZE; i++) {
#pragma unroll GLOBAL_MEM_UNROLL
                for (int j = 0; j < BLOCK_SIZE; j++) {
                    c_out[(y_block * size + x_block) * BLOCK_SIZE + j
                          + i * size] = beta * c[(y_block * size + x_block) * BLOCK_SIZE + j + i * size] +
                                  c_block[i/GEMM_BLOCK][j/GEMM_BLOCK][i & (GEMM_BLOCK - 1)][j & (GEMM_BLOCK - 1)];
                }
            }
        }
    }
}
