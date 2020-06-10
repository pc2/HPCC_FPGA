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
Calculate for the Level 2 block:

do_acc true:  c = c + a * b
do_acc false: c = a * b

where a,b,c are matrices of size GEMM_BLOCK.
Calculation itself is fully unrolled.
 */
void register_gemm(const DEVICE_DATA_TYPE a[GEMM_BLOCK][GEMM_BLOCK],
                    const DEVICE_DATA_TYPE b[GEMM_BLOCK][GEMM_BLOCK],
                    DEVICE_DATA_TYPE c_out[GEMM_BLOCK][GEMM_BLOCK],
                    const bool do_acc) {

    DEVICE_DATA_TYPE a_block[GEMM_BLOCK][GEMM_BLOCK]; // automatically in regs
    DEVICE_DATA_TYPE b_block[GEMM_BLOCK][GEMM_BLOCK]; // automatically in regs
    DEVICE_DATA_TYPE c_block[GEMM_BLOCK][GEMM_BLOCK]; // automatically in regs

    // Load block of matrix A and B from BRAM to registers
    __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
    for (int y=0; y<GEMM_BLOCK; y++) {
        __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
        for (int x=0; x<GEMM_BLOCK; x++) {
            a_block[y][x] = a[y][x];
            b_block[y][x] = b[y][x];
        }
    }

    // Calculate result for 8x8 matrix
    __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
    for (int y=0; y<GEMM_BLOCK; y++) {
        __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
        for (int x=0; x<GEMM_BLOCK; x++) {
            DEVICE_DATA_TYPE sum = 0.f;
            __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
            for (int i=0; i<GEMM_BLOCK; i++) {
                sum += a_block[y][i] * b_block[i][x];
            }
            c_block[y][x] = sum;
        }
    }

    // Write back to BRAM and accumulate
    __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
    for(int y=0; y < GEMM_BLOCK; y++) {
        __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
        for (int x=0; x<GEMM_BLOCK; x++) {
            c_out[y][x] = do_acc ? c_out[y][x] + c_block[y][x] : c_block[y][x];
        }
    }
}


/**
GEMM for the Level 1 Block (from BRAM to BRAM)

@param a_block input block from A matrix
@param b_block input block from B matrix
@param c_block result block to fill (each block will be passed in multiple times)
@param do_acc: accumulate into c_block (if false, reset to 0 at first write)
*/
void
local_gemm(const DEVICE_DATA_TYPE a_block[BLOCK_SIZE / GEMM_BLOCK][BLOCK_SIZE / GEMM_BLOCK]
                                        [GEMM_BLOCK][GEMM_BLOCK],
           const DEVICE_DATA_TYPE b_block[BLOCK_SIZE / GEMM_BLOCK][BLOCK_SIZE / GEMM_BLOCK]
                                            [GEMM_BLOCK][GEMM_BLOCK],
           DEVICE_DATA_TYPE c_block[BLOCK_SIZE / GEMM_BLOCK][BLOCK_SIZE / GEMM_BLOCK]
                                        [GEMM_BLOCK][GEMM_BLOCK],
           const bool do_acc) {
/**
The BRAM matrix multiplication works differently for Intel and Xilinx.
For Intel the kernel calculates the complete result of an GEMM_BLOCKxGEMM_BLOCK 
matrix block in registers and writes it back to BRAM. Thus, k is the most inner loop.

For Xilinx, k is the outer loop and thus it will calculate parial results for all
GEMM_BLOCKxGEMM_BLOCK matrix block and write the partial result directly back
to BRAM.
 */
#ifdef INTEL_FPGA
    #pragma loop_coalesce 2
    // For each column in top block
    for (int i = 0; i < BLOCK_SIZE / GEMM_BLOCK; i++) {
        // For each element below it in current block
        for (int j = 0; j < BLOCK_SIZE / GEMM_BLOCK; j++) {
            // For Intel FPGA accumulate all partial results in registers
            // tmp_mul and only write back to BRAM once 
            DEVICE_DATA_TYPE tmp_mul[GEMM_BLOCK][GEMM_BLOCK];
            // For each diagonal element in left block
            for (int k=0; k < BLOCK_SIZE / GEMM_BLOCK; k++) {
                // accumulate when working on following ks
                register_gemm(a_block[i][k], b_block[k][j],
                    tmp_mul, (k>0));
            }
            // Write back accumulated result to BRAM and accumulate if requested from outside
            __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
            for(int y=0; y < GEMM_BLOCK; y++) {
                __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
                for (int x=0; x<GEMM_BLOCK; x++) {
                    c_block[i][j][y][x] = do_acc ? c_block[i][j][y][x] + tmp_mul[y][x] : tmp_mul[y][x];
                }
            }    
        }
    }
#else
    // For each diagonal element in left block
    for (int k=0; k < BLOCK_SIZE / GEMM_BLOCK; k++) {
        // For each column in top block
        for (int i = 0; i < BLOCK_SIZE / GEMM_BLOCK; i++) {
            // For each element below it in current block
            for (int j = 0; j < BLOCK_SIZE / GEMM_BLOCK; j++) {
                // accumulate when requested from outside OR when working on following ks
                register_gemm(a_block[i][k], b_block[k][j],
                    c_block[i][j], do_acc | (k>0));
            }
        }
    }
#endif
}


/**
Two level blocked GEMM kernel

calculates C_OUT = alpha * A.dot(B) + beta * C

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
void gemm(__global const DEVICE_DATA_TYPE* restrict a,
          __global const DEVICE_DATA_TYPE* restrict b,
          __global const DEVICE_DATA_TYPE* restrict c,
          __global DEVICE_DATA_TYPE* restrict c_out,
          const DEVICE_DATA_TYPE alpha,
          const DEVICE_DATA_TYPE beta,
          const uint a_size) {

    const unsigned size = a_size * BLOCK_SIZE;

    // Level 1 Matrix Multiplication
#pragma loop_coalesce 2
#pragma disable_loop_pipelining
    for (int x_block = 0; x_block < a_size; x_block++) {
#pragma disable_loop_pipelining
        for (int y_block = 0; y_block < a_size; y_block++) {
            DEVICE_DATA_TYPE c_block[BLOCK_SIZE / GEMM_BLOCK][BLOCK_SIZE / GEMM_BLOCK]
            [GEMM_BLOCK][GEMM_BLOCK]  __attribute((numbanks(GEMM_BLOCK * GEMM_BLOCK),xcl_array_partition(complete, 3),xcl_array_partition(complete, 4)));

            for (int diagonal_block=0; diagonal_block < a_size; diagonal_block++) {
                DEVICE_DATA_TYPE a_block[BLOCK_SIZE / GEMM_BLOCK][BLOCK_SIZE / GEMM_BLOCK]
                                        [GEMM_BLOCK][GEMM_BLOCK]  __attribute((numbanks(GEMM_BLOCK * GEMM_BLOCK),xcl_array_partition(complete, 3),xcl_array_partition(complete, 4)));
                DEVICE_DATA_TYPE b_block[BLOCK_SIZE / GEMM_BLOCK][BLOCK_SIZE / GEMM_BLOCK]
                                        [GEMM_BLOCK][GEMM_BLOCK]  __attribute((numbanks(GEMM_BLOCK * GEMM_BLOCK),xcl_array_partition(complete, 3),xcl_array_partition(complete, 4)));
                // Load all needed level 1 blocks

#pragma loop_coalesce 2
                for (int i = 0; i < BLOCK_SIZE ; i++) {
                    for (int j = 0; j < BLOCK_SIZE; j += GLOBAL_MEM_UNROLL) {
                        DEVICE_DATA_TYPE a_reorder_buffer[GLOBAL_MEM_UNROLL];
                        DEVICE_DATA_TYPE b_reorder_buffer[GLOBAL_MEM_UNROLL];
__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
                        for (int u = 0; u < GLOBAL_MEM_UNROLL; u++) {
                            a_reorder_buffer[u] = a[(y_block * size + diagonal_block) * BLOCK_SIZE +
                                j + u + i * size];
                            b_reorder_buffer[u] = b[(diagonal_block * size + x_block) * BLOCK_SIZE +
                                                          j + u + i * size];
                        }
__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL/GEMM_BLOCK)))
                        for (int b = 0; b < GLOBAL_MEM_UNROLL/GEMM_BLOCK; b++) {
__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
                            for (int u = 0; u < GEMM_BLOCK; u++) {
                                a_block[i / GEMM_BLOCK][j / GEMM_BLOCK + b][i & (GEMM_BLOCK - 1)][u] = a_reorder_buffer[b * GEMM_BLOCK + u];
                                b_block[i / GEMM_BLOCK][j / GEMM_BLOCK + b][i & (GEMM_BLOCK - 1)][u] = b_reorder_buffer[b * GEMM_BLOCK + u];
                            }
                        }
                    }
                }

                local_gemm(a_block, b_block, c_block, diagonal_block);
            }

#pragma loop_coalesce
            for (int i = 0; i < BLOCK_SIZE; i++) {
                for (int j = 0; j < BLOCK_SIZE/GLOBAL_MEM_UNROLL; j++) {
                    DEVICE_DATA_TYPE c_reorder_buffer[GLOBAL_MEM_UNROLL];
__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
                    for (int u = 0; u < GLOBAL_MEM_UNROLL; u++) {
                        c_reorder_buffer[u] = c[(y_block * size + x_block) * BLOCK_SIZE + j * GLOBAL_MEM_UNROLL + i * size + u];
                    }
__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
                    for (int u = 0; u < GLOBAL_MEM_UNROLL; u++) {
                        c_out[(y_block * size + x_block) * BLOCK_SIZE + j * GLOBAL_MEM_UNROLL + u
                                + i * size] = beta * c_reorder_buffer[u] +
                                alpha * c_block[i/GEMM_BLOCK][(j * GLOBAL_MEM_UNROLL + u)/GEMM_BLOCK][i & (GEMM_BLOCK - 1)][(j * GLOBAL_MEM_UNROLL + u) & (GEMM_BLOCK - 1)];
                    }
                }
            }
        }
    }
}
