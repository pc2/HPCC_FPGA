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

#pragma OPENCL EXTENSION cl_intel_channels : enable

typedef struct tmp_channel_chunk { DEVICE_DATA_TYPE data[GEMM_BLOCK];} ch_chunk_t;

channel ch_chunk_t ch_top_in __attribute((io("kernel_input_ch0")));
channel ch_chunk_t ch_right_out __attribute((io("kernel_output_ch1")));
channel ch_chunk_t ch_bottom_out __attribute((io("kernel_output_ch2")));
channel ch_chunk_t ch_left_in __attribute((io("kernel_input_ch3")));

channel ch_chunk_t ch_lu_col_out;
channel ch_chunk_t ch_lu_row_out;
channel ch_chunk_t ch_top_col_in;
channel ch_chunk_t ch_top_row_out;
channel ch_chunk_t ch_left_row_in;
channel ch_chunk_t ch_left_col_out;

/**
Takes care of the external channels.
Will forward data from calculation kernels to the external channels and will forward data if required.

operation_type: 0:inner, 1: left, 2:top,3:lu
forward: if true (forward > 0), forward data to external channel, discard data otherwise. This is used to stop 
		forwarding when reaching the end of the grid
 */
__kernel
void network_layer(const uint operation_type,
				   const uint forward) {
	// For every row or column of the block, something needs to be sent
	#pragma loop_coalesce
	for (uint row = 0; row < BLOCK_SIZE; row++) {
		// Number of chunks that has to be processed
		for (uint chunk = 0; chunk < BLOCK_SIZE/GEMM_BLOCK; chunk++) {
			ch_chunk_t to_right;
			ch_chunk_t to_bottom;
			ch_chunk_t from_top;
			ch_chunk_t from_left;
			// receive extern operation
			switch (operation_type) {
				case 0: break;
				case 1: if (chunk < BLOCK_SIZE/GEMM_BLOCK - (row >> REGISTER_BLOCK_LOG)) {
							from_top = read_channel_intel(ch_top_in);
							// Forward chunk to the next top block
							to_bottom = from_top;
						}
						break;
				case 2:
				if (chunk < BLOCK_SIZE/GEMM_BLOCK - (row >> REGISTER_BLOCK_LOG)) {
					from_left = read_channel_intel(ch_left_in);
					// Forward chunk to the next top block
					to_right = from_left;
				}
					break;
				case 3: break;
			}
			// exchange intern operation
			switch (operation_type) {
				case 0: break;
				case 1: if (chunk < BLOCK_SIZE/GEMM_BLOCK - (row >> REGISTER_BLOCK_LOG)) {
							write_channel_intel(ch_left_row_in, from_top);
						}
						to_right = read_channel_intel(ch_left_col_out);
						break;
				case 2: if (chunk < BLOCK_SIZE/GEMM_BLOCK - (row >> REGISTER_BLOCK_LOG)) {
							write_channel_intel(ch_top_col_in, from_left);
						}
						to_bottom = read_channel_intel(ch_top_row_out);
						break;
				case 3: if (chunk < BLOCK_SIZE/GEMM_BLOCK - (row >> REGISTER_BLOCK_LOG)) {
							to_right = read_channel_intel(ch_lu_col_out);
							to_bottom = read_channel_intel(ch_lu_row_out);
						}
						break;
			}
			if (forward) {
				if (operation_type < 2 || chunk < BLOCK_SIZE/GEMM_BLOCK - (row >> REGISTER_BLOCK_LOG)) {
					write_channel_intel(ch_right_out, to_right);
				}
				if ((operation_type != 3 && operation_type != 1) || chunk < BLOCK_SIZE/GEMM_BLOCK - (row >> REGISTER_BLOCK_LOG)) {
					write_channel_intel(ch_bottom_out, to_bottom);
				}
			}
		}
	}
}


/**
Executes a single step of the LU factorization.

This method takes a partially solved 8x8 matrix and calculates the next step of the LU factorization
The method needs 7 (GEMM_BLOCK-1) calls to perform a single LU factorization. This is done to reduce resource usage,
since all upcomng calls are anyway depending on the results of the previous call and there is no way
to pipeline multiple executions.

A is the input block that might be partially computed
step is the current step and must be a value between 0 to GEMM_BLOCK-2. After step GEMM_BLOCK-2, the block is factorized
 */
void
lu_block(const DEVICE_DATA_TYPE A[GEMM_BLOCK][GEMM_BLOCK], const int step, DEVICE_DATA_TYPE A_out[GEMM_BLOCK][GEMM_BLOCK]) {

	// Read current line from input
	DEVICE_DATA_TYPE line[GEMM_BLOCK];
	__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
	for (int i=0; i < GEMM_BLOCK; i++) {
		line[i] = A[step][i];
	}

	// calculate the inverse of the diagonal element for the scaling
	DEVICE_DATA_TYPE inv_scale_a = -1.0 / line[step];

	// Scale the current row
	__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
	for (int i=0; i < GEMM_BLOCK; i++) {
		if (i > step) {
			line[i] = line[i] * inv_scale_a;
		}
	}
	line[step] = inv_scale_a;

	// Update all rows fully unrolled
	// The multiply adds are fully independent
	//__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
	// Unrolling disabled for this loop to save resources
	for (int j = 0; j < GEMM_BLOCK; j++) {
		DEVICE_DATA_TYPE curr_scale = A[j][step];
		// Update a single row. If it is already updated, just write back the value, if it is the current row
		// write back the value in "line", else update the value
		if (j != step) {
			__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
			for (int i = 0; i < GEMM_BLOCK; i++) {
				A_out[j][i] = (i > step && j > step) ? A[j][i] + line[i] * curr_scale : A[j][i];
			}
		}
		else {
			__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
			for (int i = 0; i < GEMM_BLOCK; i++) {
				A_out[j][i] = line[i];
			}		
		}
	}
}

/**
This function can be used to update blocks using with three different operations.
It will execute the update for a single row in the block. The update is completed after GEMM_BLOCK calls of this
update function

operation_type: 0 for top = the top row of blocks will need a triangular MM
				1 for left = the left column of blocks will need a triangular MM, matrices have to be transposed
				2 for inner block == all inner blocks will be updated with a MM
 */
void
update_block(const DEVICE_DATA_TYPE a[GEMM_BLOCK][GEMM_BLOCK], 
			 const DEVICE_DATA_TYPE top[GEMM_BLOCK],
			 const DEVICE_DATA_TYPE left_or_lu[GEMM_BLOCK],
			 DEVICE_DATA_TYPE out[GEMM_BLOCK][GEMM_BLOCK],
			 const int current_row,
			 const int operation_type) {
	
	// Define different operation types of function
	const int op_top = 0;
	const int op_left = 1;
	const int op_inner = 2;

	// Transpose the input matrices if the target is a left block
	DEVICE_DATA_TYPE current_block[GEMM_BLOCK][GEMM_BLOCK]  __attribute__((register));
	if (operation_type == op_left) {
		__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
			for (int jj =0; jj < GEMM_BLOCK; jj++) {
				current_block[ii][jj] = a[jj][ii] ;
			}
		}
	}
	else {
		__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
			for (int jj =0; jj < GEMM_BLOCK; jj++) {
				current_block[ii][jj] = a[ii][jj] ;
			}
		}
	}

	// Generate the first scalling array depending on the operation type
	DEVICE_DATA_TYPE scale_row[GEMM_BLOCK]  __attribute__((register));
	if (operation_type == op_inner) {
		__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
		for (int jj =0; jj < GEMM_BLOCK; jj++) {
			scale_row[jj] = top[jj];
		}
	}
	else {
		__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
		for (int jj =0; jj < GEMM_BLOCK; jj++) {
			scale_row[jj] = current_block[current_row][jj];
		}
	}
	if (operation_type == op_top) {
		__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
		for (int jj =0; jj < GEMM_BLOCK; jj++) {
			scale_row[jj] *= left_or_lu[current_row];
		}
	}

	DEVICE_DATA_TYPE tmp[GEMM_BLOCK][GEMM_BLOCK]  __attribute__((register));
	// scale all values with the pre calculated scaling array and the second input
	__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
	for (int ii =0; ii < GEMM_BLOCK; ii++) {
		__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
		for (int jj =0; jj < GEMM_BLOCK; jj++) {
			// left_or_lu_block are stored transposed to simplify the data access here
			tmp[ii][jj] = current_block[ii][jj] + scale_row[jj] * left_or_lu[ii];
		}
	}

	// overwrite results that were calculated altough they are not needed for the triangular operations left and top
	if (operation_type != op_inner) {
		__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			if (ii == current_row) {
				__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
				for (int jj =0; jj < GEMM_BLOCK; jj++) {
					tmp[ii][jj] = scale_row[jj];
				}
			}
			else if (ii < current_row) {
				__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
				for (int jj =0; jj < GEMM_BLOCK; jj++) {
					tmp[ii][jj] = current_block[ii][jj];
				}				
			}
		}
	}

	// write result back and transpose if necessary
	if (operation_type == op_left) {
		__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
			for (int jj =0; jj < GEMM_BLOCK; jj++) {
				out[ii][jj] = tmp[jj][ii];
			}
		}
	}
	else {
		__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
			for (int jj =0; jj < GEMM_BLOCK; jj++) {
				out[ii][jj] = tmp[ii][jj];
			}
		}		
	}
}

__attribute__((uses_global_work_offset(0)))
__kernel
void
lu(__global DEVICE_DATA_TYPE* restrict a) {

	local DEVICE_DATA_TYPE a_buffer[BLOCK_SIZE/GEMM_BLOCK][BLOCK_SIZE/GEMM_BLOCK][GEMM_BLOCK][GEMM_BLOCK];
	
	// Store current row and column in separate buffers for 
	// easier access in the deep pipeline
	// need to be declared as local to prevent the compiler from 
	local DEVICE_DATA_TYPE top_buffer[BLOCK_SIZE/GEMM_BLOCK][GEMM_BLOCK];
	local DEVICE_DATA_TYPE left_buffer[2][GEMM_BLOCK];

	// Load block to local memory
	#pragma loop_coalesce
	for (int i =0; i < BLOCK_SIZE/GEMM_BLOCK; i++) {
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			for (int j =0; j < BLOCK_SIZE/GEMM_BLOCK; j++) {
				__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
				for (int jj =0; jj < GEMM_BLOCK; jj++) {
					a_buffer[i][j][ii][jj] = a[(i * GEMM_BLOCK + ii) * BLOCK_SIZE + j * GEMM_BLOCK + jj];
				}
			}
		}
	}
	
	// For each row in the matrix update whole matrix.
	// The iterations depend on each other, so loop pipelining is disabled here
	#pragma disable_loop_pipelining
	for (int gk = 0; gk < BLOCK_SIZE; gk++) {

		int k = gk / GEMM_BLOCK;
		int kk = gk & (GEMM_BLOCK - 1);

		// Read in current LU block
		DEVICE_DATA_TYPE lu_a_buffer_in[GEMM_BLOCK][GEMM_BLOCK];
		__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
			for (int jj =0; jj < GEMM_BLOCK; jj++) {
				lu_a_buffer_in[ii][jj] = a_buffer[k][k][ii][jj];
			}
		}

		DEVICE_DATA_TYPE lu_a_buffer_out[GEMM_BLOCK][GEMM_BLOCK];
		DEVICE_DATA_TYPE lu_a_buffer_out_row[GEMM_BLOCK];
		DEVICE_DATA_TYPE lu_a_buffer_out_col[GEMM_BLOCK];
		// Calculate next row and column of LU factorization and store in local memory buffer
		lu_block(lu_a_buffer_in, kk, lu_a_buffer_out);
		__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
			for (int jj =0; jj < GEMM_BLOCK; jj++) {
				a_buffer[k][k][ii][jj] = lu_a_buffer_out[ii][jj];
			}
		}
		__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
		for (int jj =0; jj < GEMM_BLOCK; jj++) {
			lu_a_buffer_out_row[jj] = lu_a_buffer_out[kk][jj];
		}
		__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
		for (int jj =0; jj < GEMM_BLOCK; jj++) {
			lu_a_buffer_out_col[jj] = lu_a_buffer_out[jj][kk];
		}

		// The update pipeline does not need to be executed for the last
		// row of blocks
		if (gk < BLOCK_SIZE - GEMM_BLOCK) {

			// Update all other blocks with the new calculated row and column
			#pragma ivdep safelen(BLOCK_SIZE/GEMM_BLOCK)
			for (int ttj = 0; ttj < (BLOCK_SIZE/GEMM_BLOCK - k) * BLOCK_SIZE/GEMM_BLOCK; ttj++) {

				int j = (ttj) & (BLOCK_SIZE/GEMM_BLOCK - 1);
				int ti = (ttj + (k * BLOCK_SIZE/GEMM_BLOCK)) / (BLOCK_SIZE/GEMM_BLOCK);
				// always execute the pipeline for whole rows of matrix blocks.
				// Only execute update for blocks that are required.
				// This helps to keep constant latencies between data dependencies of the pipeline stages
				if (ti >= k && j >= k) {
					/*
					Update order of block is:
					First the block below the LU block
					Then the row of blocks right of LU block
					This way the left block of the next column will always be calculated in advance 
					because it will be needed as input for the subsequent blocks.
					*/
					int i = (j == k) ? ti + 1 : ti;

					// The last left block will be out of bounds because of the update strategy described above
					// Skip it
					if (i < BLOCK_SIZE/GEMM_BLOCK) {

						// TODO Split this up into three different styles to reduce logic usage?
						
						// copy the correct block in the second input buffer
						// this depends on the operations that has to be executed
						DEVICE_DATA_TYPE second_input[GEMM_BLOCK];
						if (j == k) {
							// left matrix block will be calculated
							__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
							for (int jj =0; jj < GEMM_BLOCK; jj++) {
								second_input[jj] = lu_a_buffer_out_row[jj];
							}
						}
						else if (i == k) {
							// top matrix block will be calculated
							__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
							for (int jj =0; jj < GEMM_BLOCK; jj++) {
								second_input[jj] = lu_a_buffer_out_col[jj];
							}
						}
						else {
							// inner block will be calculated
							__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
							for (int jj =0; jj < GEMM_BLOCK; jj++) {
								second_input[jj] = left_buffer[ti & 1][jj];
							}
						}
						DEVICE_DATA_TYPE a_input[GEMM_BLOCK][GEMM_BLOCK];
						__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
						for (int ii =0; ii < GEMM_BLOCK; ii++) {
							__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
							for (int jj = 0; jj < GEMM_BLOCK; jj++) {
								a_input[ii][jj] = a_buffer[i][j][ii][jj];
							}
						}
						DEVICE_DATA_TYPE top_input[GEMM_BLOCK];
						if (ttj >= BLOCK_SIZE/GEMM_BLOCK) {
							__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
							for (int jj =0; jj < GEMM_BLOCK; jj++) {
								top_input[jj] = top_buffer[j][jj];
							}
						}
						DEVICE_DATA_TYPE out[GEMM_BLOCK][GEMM_BLOCK] __attribute__((register));
						update_block(a_input, 
										top_input, 
										second_input, 
										out,
										kk,
										(i == k) ? 0 : ((j == k) ? 1 : 2));
						if (ttj < BLOCK_SIZE/GEMM_BLOCK) {
							// only update in the first row
							__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
							for (int jj =0; jj < GEMM_BLOCK; jj++) {
								top_buffer[ttj][jj] = out[kk][jj];
							}
						}
						if (i > k && j == k) {
							__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
							for (int ii =0; ii < GEMM_BLOCK; ii++) {
								left_buffer[(ti + 1) & 1][ii] = out[ii][kk];
							}
						}
						__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
						for (int ii =0; ii < GEMM_BLOCK; ii++) {
							__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
							for (int jj = 0; jj < GEMM_BLOCK; jj++) {
								a_buffer[i][j][ii][jj] = out[ii][jj];
							}
						}
					}
				}
			}
		}
		// Send current updated column to right neighbor
		for (int i = 0; i < (BLOCK_SIZE/GEMM_BLOCK - k); i++) {
			ch_chunk_t col_data;
			ch_chunk_t row_data;
			#pragma unroll GEMM_BLOCK
			for (int j = 0; j < GEMM_BLOCK; j++) {
				col_data.data[j] = a_buffer[k][i + k][kk][j];
				row_data.data[j] = a_buffer[i + k][k][j][kk];
			}
			write_channel_intel(ch_lu_col_out, col_data);
			write_channel_intel(ch_lu_row_out, row_data);
		}
 	}

	// Store block to global memory
	#pragma loop_coalesce
	for (int i =0; i < BLOCK_SIZE/GEMM_BLOCK; i++) {
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			for (int j =0; j < BLOCK_SIZE/GEMM_BLOCK; j++) {
				__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
				for (int jj =0; jj < GEMM_BLOCK; jj++) {
					a[(i * GEMM_BLOCK + ii) * BLOCK_SIZE + j * GEMM_BLOCK + jj] = a_buffer[i][j][ii][jj];
				}
			}
		}
	}
}

/**
Update the blocks to the right of the current LU block

 */
 __attribute__((uses_global_work_offset(0)))
__kernel
void top_update(__global DEVICE_DATA_TYPE* restrict a) {

	// Store current block in local memory
	local DEVICE_DATA_TYPE a_buffer[BLOCK_SIZE/GEMM_BLOCK][BLOCK_SIZE/GEMM_BLOCK][GEMM_BLOCK][GEMM_BLOCK];
	

	// Load block to local memory
	#pragma loop_coalesce
	for (int i =0; i < BLOCK_SIZE/GEMM_BLOCK; i++) {
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			for (int j =0; j < BLOCK_SIZE/GEMM_BLOCK; j++) {
				__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
				for (int jj =0; jj < GEMM_BLOCK; jj++) {
					a_buffer[i][j][ii][jj] = a[(i * GEMM_BLOCK + ii) * BLOCK_SIZE + j * GEMM_BLOCK + jj];
				}
			}
		}
	}
	
	// For each row in the matrix update whole matrix.
	// The iterations depend on each other, so loop pipelining is disabled here
	#pragma disable_loop_pipelining
	for (int gk = 0; gk < BLOCK_SIZE; gk++) {

		int k = gk / GEMM_BLOCK;
		int kk = gk & (GEMM_BLOCK - 1);

		DEVICE_DATA_TYPE current_lu_col[BLOCK_SIZE/GEMM_BLOCK][GEMM_BLOCK];
		DEVICE_DATA_TYPE current_row[BLOCK_SIZE/GEMM_BLOCK][GEMM_BLOCK];
		DEVICE_DATA_TYPE current_scale;

		for (int col = 0; col < BLOCK_SIZE / GEMM_BLOCK; col++) {
			DEVICE_DATA_TYPE scale_chunk[GEMM_BLOCK];
			ch_chunk_t col_in;
			
			// if current column data is still available read it in and store it in buffer
			if (col < BLOCK_SIZE / GEMM_BLOCK - k) {
				col_in = read_channel_intel(ch_top_col_in);
				if (col == 0) {
					current_scale = col_in.data[kk];
				}
				#pragma unroll
				for (int i =0; i < GEMM_BLOCK; i++) {
					current_lu_col[col][i] = (col > 0 || i > kk) ? col_in.data[i] : 0.f;
				}
			}
			
			// get current row chunk
			#pragma unroll
			for (int i =0; i < GEMM_BLOCK; i++) {
				scale_chunk[i] = a_buffer[k][col][kk][i];
			}

			// scale current row chunk with the rows scale factor received over the external channel
			#pragma unroll
			for (int i =0; i < GEMM_BLOCK; i++) {
				scale_chunk[i] = scale_chunk[i] * current_scale;
			}

			// Store chunk for later update and forward it over external channel
			ch_chunk_t row_out;
			#pragma unroll
			for (int i =0; i < GEMM_BLOCK; i++) {
				row_out.data[i] = scale_chunk[i];
				current_row[col][i] = scale_chunk[i];
			}
			write_channel_intel(ch_top_row_out, row_out);

			// Update local memory buffer with chunk
			#pragma unroll
			for (int i =0; i < GEMM_BLOCK; i++) {
				a_buffer[k][col][kk][i] = scale_chunk[i];
			}
		}

		// Update all remaining rows
		#pragma loop_coalesce
		for (int row = 0; row < BLOCK_SIZE/GEMM_BLOCK - k; row++) {
			// Update whole rows!
			for (int curr_col = 0; curr_col < BLOCK_SIZE/GEMM_BLOCK; curr_col++) {
				#pragma unroll
				for (int i = 0; i < GEMM_BLOCK; i++) {
					#pragma unroll
					for (int j=0; j < GEMM_BLOCK; j++) {
						a_buffer[row + k][curr_col][i][j] += current_lu_col[row][i] * current_row[curr_col][j];
					}
				}
			}
		}
 	}

	// Store block to global memory
	#pragma loop_coalesce
	for (int i =0; i < BLOCK_SIZE/GEMM_BLOCK; i++) {
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			for (int j =0; j < BLOCK_SIZE/GEMM_BLOCK; j++) {
				__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
				for (int jj =0; jj < GEMM_BLOCK; jj++) {
					a[(i * GEMM_BLOCK + ii) * BLOCK_SIZE + j * GEMM_BLOCK + jj] = a_buffer[i][j][ii][jj];
				}
			}
		}
	}
}

/**
Update the blocks below the current LU block

 */
 __attribute__((uses_global_work_offset(0)))
__kernel
void left_update(__global DEVICE_DATA_TYPE* restrict a) {

	// Store current block in local memory
	local DEVICE_DATA_TYPE a_buffer[BLOCK_SIZE/GEMM_BLOCK][BLOCK_SIZE/GEMM_BLOCK][GEMM_BLOCK][GEMM_BLOCK];

	// Load block to local memory
	#pragma loop_coalesce
	for (int i =0; i < BLOCK_SIZE/GEMM_BLOCK; i++) {
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			for (int j =0; j < BLOCK_SIZE/GEMM_BLOCK; j++) {
				__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
				for (int jj =0; jj < GEMM_BLOCK; jj++) {
					a_buffer[i][j][ii][jj] = a[(i * GEMM_BLOCK + ii) * BLOCK_SIZE + j * GEMM_BLOCK + jj];
				}
			}
		}
	}
	
	// For each row in the matrix update whole matrix.
	// The iterations depend on each other, so loop pipelining is disabled here
	#pragma disable_loop_pipelining
	for (int gk = 0; gk < BLOCK_SIZE; gk++) {

		int k = gk / GEMM_BLOCK;
		int kk = gk & (GEMM_BLOCK - 1);

		DEVICE_DATA_TYPE current_lu_row[BLOCK_SIZE/GEMM_BLOCK][GEMM_BLOCK];
		DEVICE_DATA_TYPE current_col[BLOCK_SIZE/GEMM_BLOCK][GEMM_BLOCK];
		DEVICE_DATA_TYPE current_scale;

		for (int col = 0; col < BLOCK_SIZE / GEMM_BLOCK; col++) {
			DEVICE_DATA_TYPE scale_chunk[GEMM_BLOCK];
			ch_chunk_t row_in;
			
			// if current column data is still available read it in and store it in buffer
			if (col < BLOCK_SIZE / GEMM_BLOCK - k) {
				row_in = read_channel_intel(ch_left_row_in);
				if (col == 0) {
					current_scale = row_in.data[kk];
				}
				#pragma unroll
				for (int i =0; i < GEMM_BLOCK; i++) {
					current_lu_row[col][i] = (col > 0 || i > kk) ? row_in.data[i] : 0.f;
				}
			}
			
			// get current row chunk
			#pragma unroll
			for (int i =0; i < GEMM_BLOCK; i++) {
				scale_chunk[i] = a_buffer[col][k][i][kk];
			}

			// Store chunk for later update and forward it over external channel
			ch_chunk_t col_out;
			#pragma unroll
			for (int i =0; i < GEMM_BLOCK; i++) {
				col_out.data[i] = scale_chunk[i];
				current_col[col][i] = scale_chunk[i];
			}
			write_channel_intel(ch_left_col_out, col_out);
		}

		// Update all rows
		#pragma loop_coalesce
		for (int row = 0; row < BLOCK_SIZE/GEMM_BLOCK; row++) {
			// Update only remaining row chunks
			for (int curr_col = 0; curr_col < BLOCK_SIZE/GEMM_BLOCK - k; curr_col++) {
				#pragma unroll
				for (int i = 0; i < GEMM_BLOCK; i++) {
					#pragma unroll
					for (int j=0; j < GEMM_BLOCK; j++) {
						a_buffer[row][curr_col + k][i][j] += current_lu_row[curr_col][j] * current_col[row][i];
					}
				}
			}
		}
 	}

	// Store block to global memory
	#pragma loop_coalesce
	for (int i =0; i < BLOCK_SIZE/GEMM_BLOCK; i++) {
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			for (int j =0; j < BLOCK_SIZE/GEMM_BLOCK; j++) {
				__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
				for (int jj =0; jj < GEMM_BLOCK; jj++) {
					a[(i * GEMM_BLOCK + ii) * BLOCK_SIZE + j * GEMM_BLOCK + jj] = a_buffer[i][j][ii][jj];
				}
			}
		}
	}
}


__attribute__((uses_global_work_offset(0)))
__kernel
void
gesl(__global const DEVICE_DATA_TYPE* restrict a, 
	__global DEVICE_DATA_TYPE* restrict b,
	const unsigned n_blocks) {

	const unsigned n = n_blocks * BLOCK_SIZE;

	// solve l*y = b
	// For each row in matrix

	// Go through every value in b
	#pragma disable_loop_pipelining
	for (int k = 0; k < n - 1; k++) {

		// Split the row into chunks to allow caching in local memory
		#pragma disable_loop_pipelining
		for (int row_block = k / BLOCK_SIZE; row_block < n / BLOCK_SIZE; row_block++) {
			DEVICE_DATA_TYPE b_tmp[BLOCK_SIZE];
			DEVICE_DATA_TYPE scale_b;

			// Read a chunk of b into the cache and exchange the pivot element
			for (int chunk = 0; chunk < BLOCK_SIZE/GLOBAL_MEM_UNROLL; chunk++) {
				DEVICE_DATA_TYPE b_burst[GLOBAL_MEM_UNROLL];
				__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
				for (int i = 0; i < GLOBAL_MEM_UNROLL; i++) {
					int curr_id = (row_block) * BLOCK_SIZE + chunk * GLOBAL_MEM_UNROLL + i;
					b_burst[i] = b[curr_id];
				}
				__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
				for (int i = 0; i < GLOBAL_MEM_UNROLL; i++) {
					int curr_id = (row_block) * BLOCK_SIZE + chunk * GLOBAL_MEM_UNROLL + i;
					b_tmp[chunk * GLOBAL_MEM_UNROLL + i] = b_burst[i];
					if (curr_id == k) {
						scale_b = b_burst[i];
					}
				}
			}

			// Update values of b and store them back to global memory
			for (int chunk = 0; chunk <  BLOCK_SIZE/ GLOBAL_MEM_UNROLL; chunk++) {
				DEVICE_DATA_TYPE a_burst[GLOBAL_MEM_UNROLL];
				__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
				for (int i = 0; i < GLOBAL_MEM_UNROLL; i++) {
					int curr_id = (row_block) * BLOCK_SIZE + chunk * GLOBAL_MEM_UNROLL + i;
					a_burst[i] = a[n * k + curr_id];
				}

				__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
				for (int i = 0; i < GLOBAL_MEM_UNROLL; i++) {
					int curr_id = (row_block) * BLOCK_SIZE + chunk * GLOBAL_MEM_UNROLL + i;
					DEVICE_DATA_TYPE new_val = (curr_id > k) ? b_tmp[chunk * GLOBAL_MEM_UNROLL + i] +  scale_b * a_burst[i] : b_tmp[chunk * GLOBAL_MEM_UNROLL + i];
					b[curr_id] = new_val;
				}
			}
		}
	}

	// now solve  u*x = y

	// load the current sclae value for b from global memory
	DEVICE_DATA_TYPE curr_b = b[n-1];
	DEVICE_DATA_TYPE ux_scale_b = 0.f;

	// for every value in b
	#pragma disable_loop_pipelining
	for (int k = n - 1; k >= 0; k--) {

		// Split the row into chunks to allow caching in local memory
		#pragma disable_loop_pipelining
		for (int row_block = 0; row_block <= (k >> LOCAL_MEM_BLOCK_LOG); row_block++) {
			DEVICE_DATA_TYPE b_tmp[BLOCK_SIZE];

			if (row_block == 0) {
				// scale current b value
				ux_scale_b = curr_b * a[n * k + k];
			}

			// Read a chunk of b into the cache
			for (int chunk = 0; chunk < BLOCK_SIZE/GLOBAL_MEM_UNROLL; chunk++) {
				DEVICE_DATA_TYPE b_burst[GLOBAL_MEM_UNROLL];
				__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
				for (int i = 0; i < GLOBAL_MEM_UNROLL; i++) {
					int curr_id = (row_block) * BLOCK_SIZE + chunk * GLOBAL_MEM_UNROLL + i;
					b_burst[i] = b[curr_id];
				}
				__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
				for (int i = 0; i < GLOBAL_MEM_UNROLL; i++) {
					int curr_id = (row_block) * BLOCK_SIZE + chunk * GLOBAL_MEM_UNROLL + i;
					b_tmp[chunk * GLOBAL_MEM_UNROLL + i] = b_burst[i];
				}
			}

			// scale all other values of b and write them back to global memory
			// TODO: With Vitis this pipeline has an II=16 because of non-aligned accesses
			//       to global memory(?) Why? Maybe because of ak,k loaded for scaling and the
			//		load in this pipeline? 
			#pragma nofusion
			for (int chunk = 0; chunk <  BLOCK_SIZE/GLOBAL_MEM_UNROLL; chunk++) {
				DEVICE_DATA_TYPE a_burst[GLOBAL_MEM_UNROLL];

				// read in a
				__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
				for (int i = 0; i < GLOBAL_MEM_UNROLL; i++) {
					int curr_id = (row_block) * BLOCK_SIZE + chunk * GLOBAL_MEM_UNROLL + i;
					a_burst[i] = a[n * k + curr_id];
				}

				// Update values
				__attribute__((opencl_unroll_hint(GLOBAL_MEM_UNROLL)))
				for (int i = 0; i < GLOBAL_MEM_UNROLL; i++) {
					int curr_id = (row_block) * BLOCK_SIZE + chunk * GLOBAL_MEM_UNROLL + i;
					DEVICE_DATA_TYPE new_val = (curr_id < k) ? b_tmp[chunk * GLOBAL_MEM_UNROLL + i] +  ux_scale_b * a_burst[i] : (curr_id != k) ? b_tmp[chunk * GLOBAL_MEM_UNROLL + i] : -ux_scale_b;
					b[curr_id] = new_val;
					if (curr_id == k - 1) {
						// cache next scale value for next iteration
						curr_b = new_val;
					}
				}
			}
		}
	}
}
