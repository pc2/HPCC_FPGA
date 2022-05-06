/******************************************************************************
 *  Author: Arjun Ramaswami
 *
 *  Edited by Marius Meyer:
 *  - Adapt to used kernel signature
 *  - Change to row-column loop structure
 *****************************************************************************/
#include "parameters.h"

const unsigned int block_size = BLOCK_SIZE;
const unsigned int channel_width = CHANNEL_WIDTH;



extern "C" {

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
void transpose/*PY_CODE_GEN i*/( const DEVICE_DATA_TYPE *A,
                                 const DEVICE_DATA_TYPE *B,
                                 DEVICE_DATA_TYPE *A_out,
            const unsigned int offset_a,
            const unsigned int offset_b,
            const unsigned int number_of_blocks,
            const unsigned int width_in_blocks,
            const unsigned int height_in_blocks) {

    // local memory double buffer for a matrix block
    DEVICE_DATA_TYPE a_block[block_size * block_size / channel_width][channel_width];
#pragma HLS ARRAY_PARTITION variable = a_block complete dim = 2
#pragma HLS BIND_STORAGE variable = a_block type = RAM_1P impl = URAM
    // local memory double buffer for a matrix block
    DEVICE_DATA_TYPE a_plus_b_block[block_size * block_size / channel_width][channel_width];
#pragma HLS ARRAY_PARTITION variable = a_plus_b_block complete dim = 2
#pragma HLS BIND_STORAGE variable = a_plus_b_block type = RAM_1P impl = URAM

    // transpose the matrix block-wise from global memory
block_loop:
    for (unsigned int block = 0; block < number_of_blocks; block++) {
read_A:
        for (unsigned int row = 0; row < block_size; row++) {
read_A_line:
            for (unsigned int col = 0; col < block_size / channel_width; col++) {
        #pragma HLS unroll region
                unsigned long block_row_a = (block + offset_a) / width_in_blocks;
                unsigned long block_col_a = (block + offset_a) % width_in_blocks;
                unsigned long ls_address_trans = block_col_a * block_size * block_size * height_in_blocks +
                            block_row_a * block_size + 
                            row * block_size * height_in_blocks;

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
                    a_block[chunk][unroll_count] = rotate_in[(unroll_count + channel_width - rot)
                                                                                                % (channel_width)];
                }
            }
        }

        // Read transposed A from local memory and add B 
read_B:
        for (unsigned int row = 0; row < block_size; row++) {
read_B_line:
            for (unsigned int col = 0; col < block_size / channel_width; col++) {
#pragma HLS unroll region
                unsigned long block_row = (block + offset_b) / width_in_blocks;
                unsigned long block_col = (block + offset_b) % width_in_blocks;
                unsigned long ls_address_row = block_row * block_size * block_size * width_in_blocks +
                        block_col * block_size + 
                        row * block_size * width_in_blocks;
                unsigned int chunk = row * (block_size / channel_width) + col;

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
                    rotate_out[unroll_count] = a_block[row_rotate][unroll_count];
                }

                unsigned rot_out = row % (channel_width);

                // rotate temporary buffer to store data into local buffer
                for (unsigned unroll_count = 0; unroll_count < channel_width; unroll_count++) {
                    data_chunk[unroll_count] = rotate_out[(unroll_count + rot_out) % (channel_width)];
                }

                // load tranposed A from global memory
                for (unsigned unroll_count = 0; unroll_count < channel_width; unroll_count++) {
                    data_chunk[unroll_count] += B[ls_address_row + col * channel_width + unroll_count];
                }

                for (unsigned unroll_count = 0; unroll_count < channel_width; unroll_count++) {
                    a_plus_b_block[chunk][unroll_count] = data_chunk[unroll_count];
                }
            }
        }
        // Write back result
write_result:
        for (unsigned int row = 0; row < block_size; row++) {
write_result_line:
            for (unsigned int col = 0; col < block_size / channel_width; col++) {
#pragma HLS unroll region
                unsigned long block_row = (block + offset_b) / width_in_blocks;
                unsigned long block_col = (block + offset_b) % width_in_blocks;
                unsigned long ls_address_row = block_row * block_size * block_size * width_in_blocks +
                        block_col * block_size + 
                        row * block_size * width_in_blocks;
                unsigned int chunk = row * (block_size / channel_width) + col;

                for (unsigned unroll_count = 0; unroll_count < channel_width; unroll_count++) {
                    A_out[ls_address_row + col * channel_width + unroll_count] = a_plus_b_block[chunk][unroll_count];
                }
            }
        }
    }
}

// PY_CODE_GEN block_end

}
