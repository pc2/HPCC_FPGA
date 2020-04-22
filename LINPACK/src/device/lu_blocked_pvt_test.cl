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
#include "lu_blocked_pvt.cl"


__attribute__((uses_global_work_offset(0)))
__kernel
void test_c4(global DEVICE_DATA_TYPE* restrict a, global DEVICE_DATA_TYPE* restrict b, global DEVICE_DATA_TYPE* restrict c,
        global DEVICE_DATA_TYPE* scale_factors, global int * restrict ipvt, uint a_size) {

    DEVICE_DATA_TYPE top_block_out[BLOCK_SIZE / GEMM_BLOCK][BLOCK_SIZE / GEMM_BLOCK]
    [GEMM_BLOCK][GEMM_BLOCK];
    DEVICE_DATA_TYPE left_block_out[BLOCK_SIZE / GEMM_BLOCK][BLOCK_SIZE / GEMM_BLOCK]
    [GEMM_BLOCK][GEMM_BLOCK];
    DEVICE_DATA_TYPE current_block[BLOCK_SIZE][BLOCK_SIZE];
    DEVICE_DATA_TYPE current_block_out[BLOCK_SIZE][BLOCK_SIZE];

#pragma loop_coalesce 2
    for (int i = 0; i < BLOCK_SIZE ; i++) {
        for (int j = 0; j < BLOCK_SIZE / UNROLL_COUNT; j++) {
            DEVICE_DATA_TYPE left_reorder_buffer[UNROLL_COUNT];
            DEVICE_DATA_TYPE top_reorder_buffer[UNROLL_COUNT];
#pragma unroll
            for (int u = 0; u < UNROLL_COUNT; u++) {
                left_reorder_buffer[u] = a[j * UNROLL_COUNT + u + i * a_size * BLOCK_SIZE];
                top_reorder_buffer[u] = b[j * UNROLL_COUNT + u + i * a_size * BLOCK_SIZE];
            }
#pragma unroll
            for (int k = 0; k < UNROLL_COUNT/GEMM_BLOCK; k++) {
#pragma unroll
                for (int u = 0; u < GEMM_BLOCK; u++) {
                    left_block_out[i / GEMM_BLOCK][j * (UNROLL_COUNT / GEMM_BLOCK)+ k][i & (GEMM_BLOCK - 1)][u] = left_reorder_buffer[k * GEMM_BLOCK + u];
                    top_block_out[i / GEMM_BLOCK][j * (UNROLL_COUNT / GEMM_BLOCK)+ k][i & (GEMM_BLOCK - 1)][u] = top_reorder_buffer[k * GEMM_BLOCK + u];
                }
            }
        }
    }

    load_block(current_block, c, 0,
                                    0, a_size);

    inner_blocks_c4(left_block_out, top_block_out, current_block,
                                        current_block_out);

    store_block(current_block_out, c, 0,
                                    0, a_size);


}

__attribute__((uses_global_work_offset(0)))
__kernel
void test_c2(global DEVICE_DATA_TYPE* restrict a, global DEVICE_DATA_TYPE* restrict b, global DEVICE_DATA_TYPE* restrict c,
        global DEVICE_DATA_TYPE* restrict scale_factors, global int * restrict ipvt,  uint a_size) {

    DEVICE_DATA_TYPE top_block_out[BLOCK_SIZE][BLOCK_SIZE];
    DEVICE_DATA_TYPE left_block_out[BLOCK_SIZE][BLOCK_SIZE];
    DEVICE_DATA_TYPE local_scale_factors[BLOCK_SIZE];
    DEVICE_DATA_TYPE current_block_out[BLOCK_SIZE][BLOCK_SIZE];

    for (uint i=0; i < BLOCK_SIZE; i++) {
        local_scale_factors[i] = scale_factors[i];
    }

    load_block(left_block_out, b, 0,
               0, a_size);
    load_block(top_block_out, a, 0, 0, a_size);

    left_blocks_c2(top_block_out, left_block_out,
                    current_block_out, local_scale_factors);

    store_block(current_block_out, b, 0,
                0, a_size);

}

__attribute__((uses_global_work_offset(0)))
__kernel
void test_c3(global DEVICE_DATA_TYPE* restrict a, global DEVICE_DATA_TYPE* restrict b, global DEVICE_DATA_TYPE* restrict c, global DEVICE_DATA_TYPE* restrict scale_factors,
        global int * restrict ipvt, uint a_size) {

    DEVICE_DATA_TYPE top_block_out[BLOCK_SIZE][BLOCK_SIZE];
    DEVICE_DATA_TYPE left_block_out[BLOCK_SIZE][BLOCK_SIZE];
    int local_ipvt[BLOCK_SIZE];
    DEVICE_DATA_TYPE current_block_out[BLOCK_SIZE][BLOCK_SIZE];

    for (uint i=0; i < BLOCK_SIZE; i++) {
        local_ipvt[i] = ipvt[i];
    }

    load_block(left_block_out, b, 0,
    0, a_size);
    load_block(top_block_out, a, 0, 0, a_size);

    top_blocks_c3(top_block_out, left_block_out,
            current_block_out, local_ipvt);

    store_block(current_block_out, b, 0,
    0, a_size);

}

__attribute__((uses_global_work_offset(0)))
__kernel
void test_c1(global DEVICE_DATA_TYPE* restrict a, global DEVICE_DATA_TYPE* restrict b, global DEVICE_DATA_TYPE* restrict c, global DEVICE_DATA_TYPE* restrict scale_factors,
        global int * restrict ipvt, uint a_size) {

DEVICE_DATA_TYPE top_block_out[BLOCK_SIZE][BLOCK_SIZE];
int local_ipvt[BLOCK_SIZE];
DEVICE_DATA_TYPE local_scale[BLOCK_SIZE];
DEVICE_DATA_TYPE current_block_out[BLOCK_SIZE][BLOCK_SIZE];

for (uint i=0; i < BLOCK_SIZE; i++) {
local_ipvt[i] = ipvt[i];
}

load_block(top_block_out, a, 0, 0, a_size);

lu_factorization_c1(top_block_out,
        current_block_out, local_scale, local_ipvt);

store_block(current_block_out, a, 0,
0, a_size);


}
