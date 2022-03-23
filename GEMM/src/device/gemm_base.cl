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

#if DATA_TYPE_SIZE == 8
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#if DATA_TYPE_SIZE == 2
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

// code generation expects an array of maps of size num_replications with the keys a,b,c,out.
// The value of the keys have to be strings containing the attributes that
// have to be assigned to input and output buffers in global memory
{% set kernel_param_attributes = generate_map_attributes(num_replications) %}

/**
Calculate for the Level 2 block:

do_acc true:  c = c + a * b
do_acc false: c = a * b

where a,b,c are matrices of size GEMM_BLOCK.
Calculation itself is fully unrolled.
 */
 __attribute__((always_inline))
void register_gemm(const DEVICE_DATA_TYPE a[GEMM_BLOCK][GEMM_BLOCK],
                    const DEVICE_DATA_TYPE b[GEMM_BLOCK][GEMM_BLOCK],
#ifdef INTEL_FPGA
                    DEVICE_DATA_TYPE c_out[INTEL_MUL_SHIFT_REG + 1][GEMM_BLOCK][GEMM_BLOCK],
#else
                    DEVICE_DATA_TYPE c_out[GEMM_BLOCK][GEMM_BLOCK],
#endif
                    const bool do_acc) {
#ifdef INTEL_FPGA
    /* 
     * For Intel devices do a cannon matrix multiplication. 
     * This leads to higher kernel frequencies and thus performance.
     * For Xilinx, this type of optimization does not work well, so a 
     * standard matrix multiplication is used instead
     */

    DEVICE_DATA_TYPE a_block[GEMM_BLOCK][GEMM_BLOCK + 1];
    DEVICE_DATA_TYPE b_block[GEMM_BLOCK + 1][GEMM_BLOCK];
    DEVICE_DATA_TYPE c_block[GEMM_BLOCK][GEMM_BLOCK];

    // Load block of matrix A and B and init C and reorder values
__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
    for (int y=0; y<GEMM_BLOCK; y++) {
__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
        for (int x=0; x<GEMM_BLOCK; x++) {
            int k = (x + y) % GEMM_BLOCK;
            a_block[y][x] = a[y][k];
            b_block[y][x] = b[k][x];
            c_block[y][x] = 0;
        }
    }

    // Calculate result for 8x8 matrix
    __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
    for (int i=0;i<GEMM_BLOCK; i++) {
        __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
        for (int x=0; x<GEMM_BLOCK;x++) {
            a_block[x][GEMM_BLOCK] = a_block[x][0];
            b_block[GEMM_BLOCK][x] = b_block[0][x];
        }
        __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
        for(int y=0; y < GEMM_BLOCK; y++) {
            __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
            for (int x=0; x<GEMM_BLOCK;x++) {
                c_block[y][x] += a_block[y][x] * b_block[y][x];
                a_block[y][x] = a_block[y][x + 1];
                b_block[y][x] = b_block[y + 1][x];
            }
        }
    }
    // Write back to BRAM and accumulate
    __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
    for(int y=0; y < GEMM_BLOCK; y++) {
        __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
        for (int x=0; x<GEMM_BLOCK; x++) {
            c_out[INTEL_MUL_SHIFT_REG][y][x] = c_out[0][y][x] + c_block[y][x];
        }
    }

#else
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
            DEVICE_DATA_TYPE sum = do_acc ? c_out[y][x]  : 0.f;
            __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
            for (int i=0; i<GEMM_BLOCK; i++) {
                sum += a_block[y][i] * b_block[i][x];
            }
            c_block[y][x] = sum;
        }
    }

    // Write back to BRAM
    __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
    for(int y=0; y < GEMM_BLOCK; y++) {
        __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
        for (int x=0; x<GEMM_BLOCK; x++) {
            c_out[y][x] = c_block[y][x];
        }
    }
#endif
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
            DEVICE_DATA_TYPE tmp_mul[INTEL_MUL_SHIFT_REG+1][GEMM_BLOCK][GEMM_BLOCK];
            __attribute__((opencl_unroll_hint(INTEL_MUL_SHIFT_REG + 1)))
            for (int kk = 0; kk <= INTEL_MUL_SHIFT_REG; kk++) {
                __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
                for (int ii = 0; ii < GEMM_BLOCK; ii++) {
                    __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
                    for (int jj = 0; jj < GEMM_BLOCK; jj++) {
                        tmp_mul[kk][ii][jj] = 0;
                    }
                }
            }

            // For each diagonal element in left block
            for (int k=0; k < BLOCK_SIZE / GEMM_BLOCK; k++) {
                // accumulate when working on following ks
                register_gemm(a_block[i][k], b_block[k][j],
                    tmp_mul, (k>0));
#if INTEL_MUL_SHIFT_REG > 0
                __attribute__((opencl_unroll_hint(INTEL_MUL_SHIFT_REG)))
                for (int kk = 0; kk < INTEL_MUL_SHIFT_REG; kk++) {
                    __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
                    for (int ii = 0; ii < GEMM_BLOCK; ii++) {
                        __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
                        for (int jj = 0; jj < GEMM_BLOCK; jj++) {
                            tmp_mul[kk][ii][jj] = tmp_mul[kk + 1][ii][jj];
                        }
                    }
                }  
#endif
            }
#if INTEL_MUL_SHIFT_REG > 0
            DEVICE_DATA_TYPE tmp_mul_sum[GEMM_BLOCK][GEMM_BLOCK];
            __attribute__((opencl_unroll_hint(INTEL_MUL_SHIFT_REG)))
            for (int kk = 0; kk < INTEL_MUL_SHIFT_REG; kk++) {
                __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
                for (int ii = 0; ii < GEMM_BLOCK; ii++) {
                    __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
                    for (int jj = 0; jj < GEMM_BLOCK; jj++) {
                        tmp_mul_sum[ii][jj] = (kk == 0) ? tmp_mul[kk][ii][jj] : tmp_mul_sum[ii][jj] + tmp_mul[kk][ii][jj];
                    }
                }
            } 
#endif

            // Write back accumulated result to BRAM and accumulate if requested from outside
            __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
            for(int y=0; y < GEMM_BLOCK; y++) {
                __attribute__((opencl_unroll_hint(GEMM_BLOCK)))
                for (int x=0; x<GEMM_BLOCK; x++) {
#if INTEL_MUL_SHIFT_REG > 0
                    c_block[i][j][y][x] = do_acc ? c_block[i][j][y][x] + tmp_mul_sum[y][x] : tmp_mul_sum[y][x];
#else
                    c_block[i][j][y][x] = do_acc ? c_block[i][j][y][x] + tmp_mul[0][y][x] : tmp_mul[0][y][x];
#endif
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

// Here we use the total replications. This will also create three kernels for the Xilinx compiler because they all
// use different hard-coded ranges in the outer loop
{% for i in range(num_replications) %}

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
void gemm{{ i }}(
#ifdef ENABLE_MIXED_PRECISION
        // In mixed precision convert the values accordingly 
        // from single precision to the target precision on the FPGA
            __global {{ kernel_param_attributes[i]["a"] }} const float* restrict a,
          __global {{ kernel_param_attributes[i]["b"] }} const float* restrict b,
          __global {{ kernel_param_attributes[i]["c"] }} const float* restrict c,
          __global {{ kernel_param_attributes[i]["out"] }} float* restrict c_out,
          const float alpha,
          const float beta,
#else
            __global {{ kernel_param_attributes[i]["a"] }} const DEVICE_DATA_TYPE* restrict a,
          __global {{ kernel_param_attributes[i]["b"] }} const DEVICE_DATA_TYPE* restrict b,
          __global {{ kernel_param_attributes[i]["c"] }} const DEVICE_DATA_TYPE* restrict c,
          __global {{ kernel_param_attributes[i]["out"] }} DEVICE_DATA_TYPE* restrict c_out,
          const DEVICE_DATA_TYPE alpha,
          const DEVICE_DATA_TYPE beta,
#endif
          const uint a_size,
          const uint out_offset,
          const uint max_block) {

    const unsigned size = a_size * BLOCK_SIZE;

    // Level 1 Matrix Multiplication
#ifdef INTEL_FPGA
#pragma loop_coalesce 2
#pragma disable_loop_pipelining
#endif
#ifdef XILINX_FPGA
// These two loops will not be coalesced, but should not produce much overhead because the outer loop does 
// not do a lot of iterations
#endif
    for (unsigned y_block = out_offset; y_block < max_block; y_block++) {
#ifdef INTEL_FPGA
#pragma disable_loop_pipelining
#endif
        for (unsigned x_block = 0; x_block < a_size; x_block++) {
            DEVICE_DATA_TYPE c_block[BLOCK_SIZE / GEMM_BLOCK][BLOCK_SIZE / GEMM_BLOCK]
            [GEMM_BLOCK][GEMM_BLOCK]  __attribute((numbanks(GEMM_BLOCK * GEMM_BLOCK),xcl_array_partition(complete, 3),xcl_array_partition(complete, 4)));
#ifdef INTEL_FPGA
#pragma disable_loop_pipelining
#endif
            for (unsigned diagonal_block=0; diagonal_block < a_size; diagonal_block++) {
                DEVICE_DATA_TYPE a_block[BLOCK_SIZE / GEMM_BLOCK][BLOCK_SIZE / GEMM_BLOCK]
                                        [GEMM_BLOCK][GEMM_BLOCK]  __attribute((numbanks(GEMM_BLOCK * GEMM_BLOCK),xcl_array_partition(complete, 3),xcl_array_partition(complete, 4)));
                DEVICE_DATA_TYPE b_block[BLOCK_SIZE / GEMM_BLOCK][BLOCK_SIZE / GEMM_BLOCK]
                                        [GEMM_BLOCK][GEMM_BLOCK]  __attribute((numbanks(GEMM_BLOCK * GEMM_BLOCK),xcl_array_partition(complete, 3),xcl_array_partition(complete, 4)));
                // Load all needed level 1 blocks

#ifdef INTEL_FPGA
// Coalesce both loops to generate single loop
#pragma loop_coalesce 2
#endif
#ifdef XILINX_FPGA
// pipeline the outer loop and unroll the inner loop with an II=BLOCK_SIZE/GLOBAL_MEM_UNROLL since this loop will be executed quite often.
// This heavily decreases the overall runtime of this loop. There are these two options:
// 1. pipeline outer loop and unroll inner loop. This leads to wide memory bursts over a whole block row
//        with no loop overhead between the rows. Resource usage will be increased
// 2. pipeline inner loop. Will lead to a pipeline over a single row. There will be a loop overhead for every new row,
//        so the execution time will increase. Lesser resources will be used.
//
// If the block size is getting too large, pipeline the inner loop instead
#ifdef XILINX_UNROLL_GLOBAL_MEM_PIPELINE
__attribute__((xcl_pipeline_loop(1)))
#endif
#endif
                for (unsigned i = 0; i < BLOCK_SIZE ; i++) {
#ifdef XILINX_FPGA
#ifndef XILINX_UNROLL_GLOBAL_MEM_PIPELINE
__attribute__((xcl_pipeline_loop(1)))
#endif
#endif
                    for (unsigned j = 0; j < BLOCK_SIZE; j += GLOBAL_MEM_UNROLL) {

#ifdef ENABLE_MIXED_PRECISION
                        float a_reorder_buffer[GLOBAL_MEM_UNROLL];
                        float b_reorder_buffer[GLOBAL_MEM_UNROLL];
#else
                        DEVICE_DATA_TYPE a_reorder_buffer[GLOBAL_MEM_UNROLL];
                        DEVICE_DATA_TYPE b_reorder_buffer[GLOBAL_MEM_UNROLL];
#endif
__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
                        for (unsigned u = 0; u < GLOBAL_MEM_UNROLL; u++) {
                            a_reorder_buffer[u] = a[(y_block * size + diagonal_block) * BLOCK_SIZE +
                                j + u + i * size];
                            b_reorder_buffer[u] = b[(diagonal_block * size + x_block) * BLOCK_SIZE +
                                                          j + u + i * size];
                        }
__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL/GEMM_BLOCK)))
                        for (unsigned b = 0; b < GLOBAL_MEM_UNROLL/GEMM_BLOCK; b++) {
__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
                            for (unsigned u = 0; u < GEMM_BLOCK; u++) {
#ifdef ENABLE_MIXED_PRECISION
                                vstore_half(a_reorder_buffer[b * GEMM_BLOCK + u], u, &a_block[i / GEMM_BLOCK][j / GEMM_BLOCK + b][i & (GEMM_BLOCK - 1)][0]);
                                vstore_half(b_reorder_buffer[b * GEMM_BLOCK + u], u , &b_block[i / GEMM_BLOCK][j / GEMM_BLOCK + b][i & (GEMM_BLOCK - 1)][0]);
#else
                                a_block[i / GEMM_BLOCK][j / GEMM_BLOCK + b][i & (GEMM_BLOCK - 1)][u] = a_reorder_buffer[b * GEMM_BLOCK + u];
                                b_block[i / GEMM_BLOCK][j / GEMM_BLOCK + b][i & (GEMM_BLOCK - 1)][u] = b_reorder_buffer[b * GEMM_BLOCK + u];
#endif
                            }
                        }
                    }
                }

                local_gemm(a_block, b_block, c_block, diagonal_block);
            }

    unsigned moved_y_block = y_block - out_offset;

#ifdef INTEL_FPGA
#pragma loop_coalesce
#endif
#ifdef XILINX_FPGA
// pipeline the outer loop and unroll the inner loop with an II=BLOCK_SIZE/GLOBAL_MEM_UNROLL since this loop will be executed quite often.
// This heavily decreases the overall runtime of this loop. There are these two options:
// 1. pipeline outer loop and unroll inner loop. This leads to wide memory bursts over a whole block row
//        with no loop overhead between the rows. Resource usage will be increased
// 2. pipeline inner loop. Will lead to a pipeline over a single row. There will be a loop overhead for every new row,
//        so the execution time will increase. Lesser resources will be used.
//
// If the block size is getting too large, pipeline the inner loop instead
#ifdef XILINX_UNROLL_GLOBAL_MEM_PIPELINE
__attribute__((xcl_pipeline_loop(1)))
#endif
#endif
            for (unsigned i = 0; i < BLOCK_SIZE; i++) {
#ifdef XILINX_FPGA
#ifndef XILINX_UNROLL_GLOBAL_MEM_PIPELINE
__attribute__((xcl_pipeline_loop(1)))
#endif
#endif
                for (unsigned j = 0; j < BLOCK_SIZE/GLOBAL_MEM_UNROLL; j++) {

#ifdef ENABLE_MIXED_PRECISION
                    // With half precision data type this algorithm still uses single precision for the last addition
                    // to get rid of additional conversions
                    float c_reorder_buffer[GLOBAL_MEM_UNROLL];
                    float matrix_block_part[GLOBAL_MEM_UNROLL];
                    __attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
                    for (unsigned u = 0; u < GLOBAL_MEM_UNROLL; u++) {
                        c_reorder_buffer[u] = c[(y_block * size + x_block) * BLOCK_SIZE + j * GLOBAL_MEM_UNROLL + i * size + u];
                        matrix_block_part[u] = vload_half(0, &c_block[i/GEMM_BLOCK][(j * GLOBAL_MEM_UNROLL + u)/GEMM_BLOCK][i & (GEMM_BLOCK - 1)][u & (GEMM_BLOCK - 1)]);
                    }
__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
                    for (unsigned u = 0; u < GLOBAL_MEM_UNROLL; u++) {
                        c_out[(moved_y_block * size + x_block) * BLOCK_SIZE + j * GLOBAL_MEM_UNROLL + u
                                + i * size] = beta * c_reorder_buffer[u] + alpha * c_block[i/GEMM_BLOCK][j * GLOBAL_MEM_UNROLL/ GEMM_BLOCK][i & (GEMM_BLOCK - 1)][u];
                    }
#else
                    DEVICE_DATA_TYPE c_reorder_buffer[GLOBAL_MEM_UNROLL];
__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
                    for (unsigned u = 0; u < GLOBAL_MEM_UNROLL; u++) {
                        c_reorder_buffer[u] = c[(y_block * size + x_block) * BLOCK_SIZE + j * GLOBAL_MEM_UNROLL + i * size + u];
                    }
__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
                    for (unsigned u = 0; u < GLOBAL_MEM_UNROLL; u++) {
                        c_out[(moved_y_block * size + x_block) * BLOCK_SIZE + j * GLOBAL_MEM_UNROLL + u
                                + i * size] = beta * c_reorder_buffer[u] +
                                alpha * c_block[i/GEMM_BLOCK][(j * GLOBAL_MEM_UNROLL + u)/GEMM_BLOCK][i & (GEMM_BLOCK - 1)][(j * GLOBAL_MEM_UNROLL + u) & (GEMM_BLOCK - 1)];
                    }
#endif
                }
            }
        }
    }
}

{% endfor %}
