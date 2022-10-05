/******************************************************************************
 *  Author: Arjun Ramaswami
 *
 *  Edited by Marius Meyer:
 *  - Adapt to used kernel signature
 *  - Change to row-column loop structure
 *****************************************************************************/
#include "parameters.h"
#include "ap_int.h"
#include "ap_utils.h"
#include "ap_axi_sdata.h"
#include "accl_hls.h"

const unsigned int block_size = BLOCK_SIZE;
const unsigned int channel_width = CHANNEL_WIDTH;

// PY_CODE_GEN block_start [replace(local_variables=locals()) for i in range(num_replications)]

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
 * @param offset Offset in blocks that is used to read the current block of A. Since A is read column-wise
                on the block level, the whole matrix A might be written to global memory and the relevant columns
                need to be picked using this offset.
 * @param number_of_blocks The number of blocks that will be processed starting from the block offset
 * @param width_in_blocks The with of matrix A in blocks
 * @param height_in_blocks The height of matix A in blocks
 */
void transpose_read/*PY_CODE_GEN i*/( const DEVICE_DATA_TYPE *A,
            const unsigned int offset_a,
            const unsigned int number_of_blocks,
            const unsigned int width_in_blocks,
            const unsigned int height_in_blocks,
            STREAM<stream_word> &krnl2cclo) {
#pragma HLS INTERFACE axis register both port=krnl2cclo

    // local memory double buffer for a matrix block
    DEVICE_DATA_TYPE a_block[2][block_size * block_size / channel_width][channel_width];
#pragma HLS ARRAY_PARTITION variable = a_block complete dim = 3

    // transpose the matrix block-wise from global memory
block_loop:
    for (unsigned int block = 0; block < number_of_blocks + 1; block++) {
#pragma HLS loop_tripcount min=1 max=1024 avg=1

read_A:
        for (unsigned int row = 0; row < block_size; row++) {
read_A_line:
            for (unsigned int col = 0; col < block_size / channel_width; col++) {
#pragma HLS PIPELINE
                unsigned long block_row_a = (block + offset_a) / width_in_blocks;
                unsigned long block_col_a = (block + offset_a) % width_in_blocks;
                unsigned long ls_address_trans = block_col_a * block_size * block_size * height_in_blocks +
                            block_row_a * block_size + 
                            row * block_size * height_in_blocks;

#ifdef EMULATE
                // This condition is actually required to not read out of bounds
                // but prevents memory bursts, so for hardware this should be removed
                // In emulation it prevents segfaults
                if (block < number_of_blocks) {
#endif
                    // read in block of A from global memory and store it in a memory efficient manner for transpose
                    DEVICE_DATA_TYPE rotate_in[channel_width];
#pragma HLS ARRAY_PARTITION variable = rotate_in complete dim = 0

                    // Blocks of a will be stored columnwise in global memory
                    for (unsigned unroll_count = 0; unroll_count < channel_width; unroll_count++) {
                        rotate_in[unroll_count] = A[ls_address_trans + col * channel_width + unroll_count];
                    }

                    unsigned int chunk = row * (block_size / channel_width) + col;

                    unsigned rot = (row) % (channel_width);

                    // rotate temporary buffer to store data into local buffer
                    for (unsigned unroll_count = 0; unroll_count < channel_width; unroll_count++) {
                        // every block of (N / channel_width), rotates the index by 1
                        // store in double buffer
                        a_block[block & 1][chunk][unroll_count] = rotate_in[(unroll_count + channel_width - rot)
                                                                                                    % (channel_width)];
                    }
#ifdef EMULATE
                }
#endif
                if (block > 0) {
                    DEVICE_DATA_TYPE data_chunk[channel_width];
#pragma HLS ARRAY_PARTITION variable = data_chunk complete dim = 0
                    DEVICE_DATA_TYPE rotate_out[channel_width];
#pragma HLS ARRAY_PARTITION variable = rotate_out complete dim = 0

                    unsigned int base = col * block_size;
                    unsigned int offset = row / channel_width;

                    for (unsigned unroll_count = 0; unroll_count < channel_width; unroll_count++) {
                        unsigned rot = ((channel_width + unroll_count - row) * (block_size / channel_width)) %
                                                                                                    (block_size);
                        unsigned row_rotate = base + offset + rot;
                        rotate_out[unroll_count] = a_block[(block - 1) & 1][row_rotate][unroll_count];
                    }

                    unsigned rot_out = row % (channel_width);

                    // rotate temporary buffer to store data into local buffer
                    for (unsigned unroll_count = 0; unroll_count < channel_width; unroll_count++) {
                        data_chunk[unroll_count] = rotate_out[(unroll_count + rot_out) % (channel_width)];
                    }

                    stream_word tmp;

                    // load tranposed A from global memory
                    for (unsigned unroll_count = 0; unroll_count < channel_width; unroll_count++) {
                        DEVICE_DATA_TYPE v = data_chunk[unroll_count];
                        tmp.data((unroll_count + 1) * sizeof(DEVICE_DATA_TYPE)*8 - 1, unroll_count * sizeof(DEVICE_DATA_TYPE)*8) 
                                = *reinterpret_cast<ap_uint<sizeof(DEVICE_DATA_TYPE)*8>*>(&v);
                    }
                    tmp.dest = 9;
                    tmp.last = 1;
                    tmp.keep = -1;
                    STREAM_WRITE(krnl2cclo,tmp);              
                }
            }
        }
    }
}

/**
 *
 * ext. channel -> trans(A) + B -> A_out
 *
 * @param B Buffer for matrix B
 * @param A_out Buffer for result matrix
 * @param offset Offset in blocks that is used to read the current block of A. Since A is read column-wise
                on the block level, the whole matrix A might be written to global memory and the relevant columns
                need to be picked using this offset.
 * @param number_of_blocks The number of blocks that will be processed starting from the block offset
 * @param width_in_blocks The with of matrix A in blocks
 * @param height_in_blocks The height of matix A in blocks
 */
void transpose_write/*PY_CODE_GEN i*/(const DEVICE_DATA_TYPE *B,
                                 DEVICE_DATA_TYPE *A_out,
            const unsigned int offset_b,
            const unsigned int number_of_blocks,
            const unsigned int width_in_blocks,
            const unsigned int height_in_blocks,
            STREAM<stream_word> &cclo2krnl) {
#pragma HLS INTERFACE axis register both port=cclo2krnl

    // transpose the matrix block-wise from global memory
block_loop:
    for (unsigned int block = 0; block < number_of_blocks; block++) {
#pragma HLS loop_tripcount min=1 max=1024 avg=1
        // Read transposed A from local memory and add B 
read_B:
        for (unsigned int row = 0; row < block_size; row++) {
read_B_line:
            for (unsigned int col = 0; col < block_size / channel_width; col++) {
                unsigned long block_row = (block + offset_b) / width_in_blocks;
                unsigned long block_col = (block + offset_b) % width_in_blocks;
                unsigned long ls_address_row = block_row * block_size * block_size * width_in_blocks +
                        block_col * block_size + 
                        row * block_size * width_in_blocks;
                unsigned int chunk = row * (block_size / channel_width) + col;

                DEVICE_DATA_TYPE data_chunk[channel_width];
#pragma HLS ARRAY_PARTITION variable = data_chunk complete dim = 0

                stream_word tmp = STREAM_READ(cclo2krnl);

                // rotate temporary buffer to store data into local buffer
                for (unsigned unroll_count = 0; unroll_count < channel_width; unroll_count++) {
                    ap_uint<sizeof(DEVICE_DATA_TYPE)*8> v = tmp.data((unroll_count + 1) * sizeof(DEVICE_DATA_TYPE)*8 - 1, unroll_count * sizeof(DEVICE_DATA_TYPE)*8);
                    data_chunk[unroll_count] = *reinterpret_cast<DEVICE_DATA_TYPE*>(&v);
                }

                // load tranposed A from global memory
                for (unsigned unroll_count = 0; unroll_count < channel_width; unroll_count++) {
                    data_chunk[unroll_count] += B[ls_address_row + col * channel_width + unroll_count];
                }

                for (unsigned unroll_count = 0; unroll_count < channel_width; unroll_count++) {
                    A_out[ls_address_row + col * channel_width + unroll_count] = data_chunk[unroll_count];
                }
            }
        }
    }
}

// PY_CODE_GEN block_end

