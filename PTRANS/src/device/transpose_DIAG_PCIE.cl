/******************************************************************************
 *  Author: Arjun Ramaswami
 *
 *  Edited by Marius Meyer:
 *  - Adapt to used kernel signature
 *  - Change to row-column loop structure
 *****************************************************************************/

#include "parameters.h"

/**
* Load a block of A into local memory in a reordered fashion
* to transpose it half-way
*
*
* @param A Buffer for matrix A
* @param local_buffer The local memory buffer the block is stored into
* @param current_block Index of the current block used to calculate the offset in global memory
*
*/
void
load_chunk_of_a(__global DEVICE_DATA_TYPE *restrict A,
        DEVICE_DATA_TYPE local_buffer[BLOCK_SIZE * BLOCK_SIZE / CHANNEL_WIDTH][CHANNEL_WIDTH],
        const ulong current_block,
        const ulong chunk) {

        ulong local_mem_converted_row = chunk;

        DEVICE_DATA_TYPE rotate_in[CHANNEL_WIDTH];

        ulong load_address = current_block * BLOCK_SIZE * BLOCK_SIZE + chunk * CHANNEL_WIDTH;

        // Blocks of a will be stored columnwise in global memory
__attribute__((opencl_unroll_hint(CHANNEL_WIDTH)))
        for (unsigned unroll_count = 0; unroll_count < CHANNEL_WIDTH; unroll_count++) {
            rotate_in[unroll_count] = A[load_address + unroll_count];
        }

        unsigned rot = (chunk / (BLOCK_SIZE / CHANNEL_WIDTH)) & (CHANNEL_WIDTH - 1);

        // rotate temporary buffer to store data into local buffer
__attribute__((opencl_unroll_hint(CHANNEL_WIDTH)))
        for (unsigned unroll_count = 0; unroll_count < CHANNEL_WIDTH; unroll_count++) {
            // every block of (N / CHANNEL_WIDTH), rotates the index by 1
            // store in double buffer
            local_buffer[local_mem_converted_row][unroll_count] = rotate_in[(unroll_count + CHANNEL_WIDTH - rot)
                                                                                        & (CHANNEL_WIDTH - 1)];
        }
}

/**
* send a chunk of A into local memory in a reordered fashion
* to transpose it half-way
*
*
* @param A Buffer for matrix A
* @param local_buffer The local memory buffer the block is stored into
* @param current_block Index of the current block used to calculate the offset in global memory
*
*/
void
load_chunk_of_trans_a(const DEVICE_DATA_TYPE local_buffer[BLOCK_SIZE * BLOCK_SIZE / CHANNEL_WIDTH][CHANNEL_WIDTH],
    DEVICE_DATA_TYPE chunk_out[CHANNEL_WIDTH],
    const ulong chunk) {

        DEVICE_DATA_TYPE rotate_out[CHANNEL_WIDTH];

        ulong base = (chunk & (BLOCK_SIZE / CHANNEL_WIDTH - 1)) * BLOCK_SIZE;
        ulong offset = (chunk / (BLOCK_SIZE / CHANNEL_WIDTH)) / CHANNEL_WIDTH;


__attribute__((opencl_unroll_hint(CHANNEL_WIDTH)))
        for (unsigned unroll_count = 0; unroll_count < CHANNEL_WIDTH; unroll_count++) {
            unsigned rot = ((CHANNEL_WIDTH + unroll_count - (chunk / (BLOCK_SIZE / CHANNEL_WIDTH))) * (BLOCK_SIZE / CHANNEL_WIDTH)) &
                                                                                        (BLOCK_SIZE - 1);
            unsigned row_rotate = base + offset + rot;
            rotate_out[unroll_count] = local_buffer[row_rotate][unroll_count];
        }

        unsigned rot_out = (chunk / (BLOCK_SIZE / CHANNEL_WIDTH)) & (CHANNEL_WIDTH - 1);

        // rotate temporary buffer to store data into local buffer
        __attribute__((opencl_unroll_hint(CHANNEL_WIDTH)))
        for (unsigned unroll_count = 0; unroll_count < CHANNEL_WIDTH; unroll_count++) {
            chunk_out[unroll_count] = rotate_out[(unroll_count + rot_out) & (CHANNEL_WIDTH - 1)];
        }

        
}

void
add_a_and_b(__global DEVICE_DATA_TYPE *restrict B,
    const DEVICE_DATA_TYPE local_buffer_a[BLOCK_SIZE * BLOCK_SIZE / CHANNEL_WIDTH][CHANNEL_WIDTH],
    DEVICE_DATA_TYPE local_buffer_a_plus_b[BLOCK_SIZE * BLOCK_SIZE / CHANNEL_WIDTH][CHANNEL_WIDTH],
    const ulong block,
    const ulong chunk) {

    DEVICE_DATA_TYPE data_chunk[CHANNEL_WIDTH];

    load_chunk_of_trans_a(local_buffer_a, data_chunk, chunk);

    ulong ls_address = block * BLOCK_SIZE * BLOCK_SIZE + chunk * CHANNEL_WIDTH;
    // load tranposed A from global memory
    __attribute__((opencl_unroll_hint(CHANNEL_WIDTH)))
    for (unsigned unroll_count = 0; unroll_count < CHANNEL_WIDTH; unroll_count++) {
        data_chunk[unroll_count] += B[ls_address + unroll_count];
    }

    __attribute__((opencl_unroll_hint(CHANNEL_WIDTH)))
    for (unsigned unroll_count = 0; unroll_count < CHANNEL_WIDTH; unroll_count++) {
        local_buffer_a_plus_b[chunk][unroll_count] = data_chunk[unroll_count];
    }
}


void
store_a(__global DEVICE_DATA_TYPE *restrict A_out,
    const DEVICE_DATA_TYPE local_buffer_a_plus_b[BLOCK_SIZE * BLOCK_SIZE / CHANNEL_WIDTH][CHANNEL_WIDTH],
    const ulong block,
    const ulong chunk) {

    ulong ls_address = block * BLOCK_SIZE * BLOCK_SIZE + chunk * CHANNEL_WIDTH;

    __attribute__((opencl_unroll_hint(CHANNEL_WIDTH)))
    for (unsigned unroll_count = 0; unroll_count < CHANNEL_WIDTH; unroll_count++) {
        A_out[ls_address + unroll_count] = local_buffer_a_plus_b[chunk][unroll_count];
    }
}

{% for i in range(num_replications) %}

/**
 * Read blocks of matrix A and transpose them in memory.
 * Write the block into an external channel.
 *
 * Will do the following:
 *
 * A -> trans(A) -> ext. ch
 *
 * @param A Buffer for matrix A
 * @param B Buffer for matrix B
 * @param A_out Buffer for result matrix
 * @param number_of_blocks The number of blocks that will be processed starting from the block offset
 */
__attribute__((max_global_work_dim(0)))
__kernel
void transpose{{ i }}(__global DEVICE_DATA_TYPE *restrict A,
                                __global DEVICE_DATA_TYPE *restrict B,
                                __global DEVICE_DATA_TYPE *restrict A_out,
            const uint number_of_blocks) {

    // local memory double buffer for a matrix block
    DEVICE_DATA_TYPE a_block[BLOCK_SIZE * BLOCK_SIZE / CHANNEL_WIDTH][CHANNEL_WIDTH] __attribute__((xcl_array_partition(cyclic, CHANNEL_WIDTH,1))) __attribute__((xcl_array_partition(cyclic, CHANNEL_WIDTH,2)));
    // local memory double buffer for a matrix block
    DEVICE_DATA_TYPE a_plus_b_block[BLOCK_SIZE * BLOCK_SIZE / CHANNEL_WIDTH][CHANNEL_WIDTH] __attribute__((xcl_array_partition(cyclic, CHANNEL_WIDTH,1))) __attribute__((xcl_array_partition(cyclic, CHANNEL_WIDTH,2)));

    // transpose the matrix block-wise from global memory
    #pragma loop_coalesce
    for (uint block = 0; block < number_of_blocks; block++) {
        // Combine all three steps in single pipeline to reduce overhead
        for (uint chunk = 0; chunk < 3 * BLOCK_SIZE * BLOCK_SIZE / CHANNEL_WIDTH; chunk++) {
            uint current_chunk = chunk & (BLOCK_SIZE * BLOCK_SIZE / CHANNEL_WIDTH - 1);
            switch (chunk / (BLOCK_SIZE * BLOCK_SIZE / CHANNEL_WIDTH)) {
                // read in block of A from global memory and store it in a memory efficient manner for transpose
                case 0: load_chunk_of_a(A, a_block, block, current_chunk); break;
                // read transposed block of A from local memory buffer and add B from global memory to it
                case 1: add_a_and_b(B, a_block, a_plus_b_block, block, current_chunk); break;
                // Store result in global memory
                case 2: store_a(A_out, a_plus_b_block, block, current_chunk); break;
            }
        }
    }
}

{% endfor %}
