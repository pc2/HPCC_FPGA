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

// Current implementation uses __fpga_reg call to add additional registers for 
#ifdef XILINX_FPGA
#define __fpga_reg(x) x
#endif

#define BLOCK_SIZE (1 << LOCAL_MEM_BLOCK_LOG)
#define GEMM_BLOCK (1 << REGISTER_BLOCK_LOG)
#define GEMM_BLOCK_MM (1 << REGISTER_BLOCK_MM_LOG)

#ifdef INTEL_FPGA
#pragma OPENCL EXTENSION cl_intel_channels : enable
#endif

typedef struct tmp_channel_chunk { DEVICE_DATA_TYPE data[GEMM_BLOCK];} ch_chunk_t;

/**
Executes a single step of the LU factorization.

This method takes a partially solved 8x8 matrix and calculates the next step of the LU factorization
The method needs 7 (GEMM_BLOCK-1) calls to perform a single LU factorization. This is done to reduce resource usage,
since all upcomng calls are anyway depending on the results of the previous call and there is no way
to pipeline multiple executions.

A is the input block that might be partially computed
step is the current step and must be a value between 0 to GEMM_BLOCK-2. After step GEMM_BLOCK-2, the block is factorized
 */
  __attribute__((always_inline))
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
	DEVICE_DATA_TYPE current_block[GEMM_BLOCK][GEMM_BLOCK]  __attribute__((register, xcl_array_partition(complete, 1), xcl_array_partition(complete, 2)));
	if (operation_type == op_left) {
		__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
			for (int jj =0; jj < GEMM_BLOCK; jj++) {
				current_block[ii][jj] = __fpga_reg(a[jj][ii]);
			}
		}
	}
	else {
		__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
			for (int jj =0; jj < GEMM_BLOCK; jj++) {
				current_block[ii][jj] = __fpga_reg(a[ii][jj]);
			}
		}
	}

	// Generate the first scalling array depending on the operation type
	DEVICE_DATA_TYPE scale_row[GEMM_BLOCK]  __attribute__((register, xcl_array_partition(complete, 1)));
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

	DEVICE_DATA_TYPE tmp[GEMM_BLOCK][GEMM_BLOCK]  __attribute__((register, xcl_array_partition(complete, 1), xcl_array_partition(complete, 2)));
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
				out[ii][jj] = __fpga_reg(tmp[jj][ii]);
			}
		}
	}
	else {
		__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
			for (int jj =0; jj < GEMM_BLOCK; jj++) {
				out[ii][jj] = __fpga_reg(tmp[ii][jj]);
			}
		}		
	}
}

__attribute__((uses_global_work_offset(0)))
__kernel
void
lu(__global DEVICE_DATA_TYPE* restrict a, 
   __global DEVICE_DATA_TYPE* restrict a_block_trans,
   __global DEVICE_DATA_TYPE* restrict a_block,
				const uint block_col,
				const uint block_row,
				const uint blocks_per_row) {

	local DEVICE_DATA_TYPE a_buffer[BLOCK_SIZE/GEMM_BLOCK][BLOCK_SIZE/GEMM_BLOCK][GEMM_BLOCK][GEMM_BLOCK] __attribute((xcl_array_partition(complete, 3),xcl_array_partition(complete, 4)));
	
	// Store current row and column in separate buffers for 
	// easier access in the deep pipeline
	// need to be declared as local to prevent the compiler from 
	local DEVICE_DATA_TYPE top_buffer[BLOCK_SIZE/GEMM_BLOCK][GEMM_BLOCK] __attribute((xcl_array_partition(complete, 2)));
	local DEVICE_DATA_TYPE left_buffer[BLOCK_SIZE/GEMM_BLOCK][GEMM_BLOCK] __attribute((xcl_array_partition(complete, 2)));

	// Load block to local memory
	#pragma loop_coalesce
	for (int i =0; i < BLOCK_SIZE/GEMM_BLOCK; i++) {
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			for (int j =0; j < BLOCK_SIZE/GEMM_BLOCK; j++) {
				__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
				for (int jj =0; jj < GEMM_BLOCK; jj++) {
					a_buffer[i][j][ii][jj] = a[block_col * BLOCK_SIZE  + (block_row * BLOCK_SIZE + i * GEMM_BLOCK + ii) * BLOCK_SIZE * blocks_per_row + j * GEMM_BLOCK + jj];
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

		DEVICE_DATA_TYPE lu_a_buffer_out[GEMM_BLOCK][GEMM_BLOCK] __attribute((xcl_array_partition(complete, 1),xcl_array_partition(complete, 2)));
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

			// update all left blocks
			for (int tj = 1; tj < BLOCK_SIZE/GEMM_BLOCK; tj++) {

				int j = k;
				int i = tj;
				
				if (i > k) {
					// copy the correct block in the second input buffer
					// this depends on the operations that has to be executed
					DEVICE_DATA_TYPE second_input[GEMM_BLOCK];

					// left matrix block will be calculated
					__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
					for (int jj =0; jj < GEMM_BLOCK; jj++) {
						second_input[jj] = __fpga_reg(lu_a_buffer_out_row[jj]);
					}
					DEVICE_DATA_TYPE a_input[GEMM_BLOCK][GEMM_BLOCK] __attribute__((xcl_array_partition(complete, 1),xcl_array_partition(complete, 2)));
					__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
					for (int ii =0; ii < GEMM_BLOCK; ii++) {
						__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
						for (int jj = 0; jj < GEMM_BLOCK; jj++) {
							a_input[ii][jj] = __fpga_reg(a_buffer[i][j][ii][jj]);
						}
					}
					DEVICE_DATA_TYPE top_input[GEMM_BLOCK];
					DEVICE_DATA_TYPE out[GEMM_BLOCK][GEMM_BLOCK] __attribute__((register, xcl_array_partition(complete, 1), xcl_array_partition(complete, 2)));
					update_block(a_input, 
									top_input, 
									second_input, 
									out,
									kk,
									1);

					__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
					for (int ii =0; ii < GEMM_BLOCK; ii++) {
						left_buffer[i][ii] = __fpga_reg(out[ii][kk]);
					}
					__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
					for (int ii =0; ii < GEMM_BLOCK; ii++) {
						__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
						for (int jj = 0; jj < GEMM_BLOCK; jj++) {
							a_buffer[i][j][ii][jj] = __fpga_reg(out[ii][jj]);
						}
					}
				}
			}

			// Update all other blocks with the new calculated row and column
			// First update top blocks, then update left blocks, then all inner blocks
			// ti == 0: top blocks
			// ti == 1: left blocks
			// ti > 1: inner blocks
#ifdef INTEL_FPGA
			#pragma loop_coalesce
			#pragma ivdep safelen(BLOCK_SIZE/GEMM_BLOCK - 1)
#endif
			for (int ti = 0; ti < BLOCK_SIZE/GEMM_BLOCK - k; ti++) {
#ifdef INTEL_FPGA
				#pragma ivdep
#endif
				for (int tj = 1; tj < BLOCK_SIZE/GEMM_BLOCK; tj++) {

					int j = tj;
					int i = ti + k;
					// always execute the pipeline for whole rows of matrix blocks.
					// Only execute update for blocks that are required.
					// This helps to keep constant latencies between data dependencies of the pipeline stages
					if ((i > k || ti == 0) && j > k ) {
						
						// copy the correct block in the second input buffer
						// this depends on the operations that has to be executed
						DEVICE_DATA_TYPE second_input[GEMM_BLOCK];
						if (ti == 0) {
							// top matrix block will be calculated
							__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
							for (int jj =0; jj < GEMM_BLOCK; jj++) {
								second_input[jj] = __fpga_reg(lu_a_buffer_out_col[jj]);
							}
						}
						else {
							// inner block will be calculated
							__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
							for (int jj =0; jj < GEMM_BLOCK; jj++) {
								second_input[jj] = __fpga_reg(left_buffer[i][jj]);
							}
						}
						DEVICE_DATA_TYPE a_input[GEMM_BLOCK][GEMM_BLOCK] __attribute__((xcl_array_partition(complete, 1),xcl_array_partition(complete, 2)));
						__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
						for (int ii =0; ii < GEMM_BLOCK; ii++) {
							__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
							for (int jj = 0; jj < GEMM_BLOCK; jj++) {
								a_input[ii][jj] = __fpga_reg(a_buffer[i][j][ii][jj]);
							}
						}
						DEVICE_DATA_TYPE top_input[GEMM_BLOCK];
						__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
						for (int jj =0; jj < GEMM_BLOCK; jj++) {
							top_input[jj] = __fpga_reg(top_buffer[j][jj]);
						}
						DEVICE_DATA_TYPE out[GEMM_BLOCK][GEMM_BLOCK] __attribute__((register, xcl_array_partition(complete, 1), xcl_array_partition(complete, 2)));
						update_block(a_input, 
										top_input, 
										second_input, 
										out,
										kk,
										(ti == 0) ? 0 : 2);
						if (ti == 0) {
							// only update in the first row
							__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
							for (int jj =0; jj < GEMM_BLOCK; jj++) {
								top_buffer[j][jj] = __fpga_reg(out[kk][jj]);
							}
						}
						__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
						for (int ii =0; ii < GEMM_BLOCK; ii++) {
							__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
							for (int jj = 0; jj < GEMM_BLOCK; jj++) {
								a_buffer[i][j][ii][jj] = __fpga_reg(out[ii][jj]);
							}
						}
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
				__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
				for (int jj =0; jj < GEMM_BLOCK; jj++) {
					a[block_col * BLOCK_SIZE  + (block_row * BLOCK_SIZE + i * GEMM_BLOCK + ii) * BLOCK_SIZE * blocks_per_row + j * GEMM_BLOCK + jj] = a_buffer[i][j][ii][jj];
				}
			}
		}
	}
	// Store current block in global memory also transposed to allow easier access from the top kernel
	#pragma loop_coalesce
	for (int i =0; i < BLOCK_SIZE/GEMM_BLOCK; i++) {
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			for (int j =0; j < BLOCK_SIZE/GEMM_BLOCK; j++) {
				__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
				for (int jj =0; jj < GEMM_BLOCK; jj++) {
					a_block_trans[(i * GEMM_BLOCK + ii) * BLOCK_SIZE + j * GEMM_BLOCK + jj] = a_buffer[j][i][jj][ii];
				}
			}
		}
	}
	#pragma loop_coalesce
	for (int i =0; i < BLOCK_SIZE/GEMM_BLOCK; i++) {
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			for (int j =0; j < BLOCK_SIZE/GEMM_BLOCK; j++) {
				__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
				for (int jj =0; jj < GEMM_BLOCK; jj++) {
					a_block[(i * GEMM_BLOCK + ii) * BLOCK_SIZE + j * GEMM_BLOCK + jj] = a_buffer[i][j][ii][jj];
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
void top_update(__global DEVICE_DATA_TYPE* restrict a, 
				__global DEVICE_DATA_TYPE* restrict top_block, 
				__global DEVICE_DATA_TYPE* restrict lu_global_buffer_transposed,
				const uint is_first_block,
				const uint block_col,
				const uint block_row,
				const uint blocks_per_row) {

	// Store current block in local memory
	local DEVICE_DATA_TYPE a_buffer[BLOCK_SIZE/GEMM_BLOCK][BLOCK_SIZE/GEMM_BLOCK][GEMM_BLOCK][GEMM_BLOCK] __attribute((xcl_array_partition(complete, 3),xcl_array_partition(complete, 4)));
	

	// Load block to local memory
	#pragma loop_coalesce
	for (int i =0; i < BLOCK_SIZE/GEMM_BLOCK; i++) {
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			for (int j =0; j < BLOCK_SIZE/GEMM_BLOCK; j++) {
				__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
				for (int jj =0; jj < GEMM_BLOCK; jj++) {
					a_buffer[i][j][ii][jj] = a[block_col * BLOCK_SIZE  + (block_row * BLOCK_SIZE + i * GEMM_BLOCK + ii) * BLOCK_SIZE * blocks_per_row + j * GEMM_BLOCK + jj];
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

		DEVICE_DATA_TYPE current_lu_col[BLOCK_SIZE/GEMM_BLOCK][GEMM_BLOCK] __attribute((xcl_array_partition(complete, 2)));
		DEVICE_DATA_TYPE current_row[BLOCK_SIZE/GEMM_BLOCK][GEMM_BLOCK] __attribute((xcl_array_partition(complete, 2)));
		DEVICE_DATA_TYPE current_scale;

		for (int col = 0; col < BLOCK_SIZE / GEMM_BLOCK; col++) {
			ch_chunk_t col_in;

			DEVICE_DATA_TYPE scale_chunk[GEMM_BLOCK] __attribute((xcl_array_partition(complete, 1)));

			// get current row chunk
			__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
			for (int i =0; i < GEMM_BLOCK; i++) {
				scale_chunk[i] = a_buffer[k][col][kk][i];
			}
			
			// if current column data is still available read it in and store it in buffer
			if (col < BLOCK_SIZE / GEMM_BLOCK - k) {
				// Load LU data from global memory instead of receiving it from the channel
				__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
				for (int i=0; i < GEMM_BLOCK; i++) {
					col_in.data[i] = lu_global_buffer_transposed[gk * BLOCK_SIZE + (col + k) * GEMM_BLOCK + i];
				}
				if (col == 0) {
					current_scale = col_in.data[kk];
				}
				__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
				for (int i =0; i < GEMM_BLOCK; i++) {
					current_lu_col[col][i] = (col > 0 || i > kk) ? col_in.data[i] : 0.f;
				}
			}

			// scale current row chunk with the rows scale factor received over the external channel
			__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
			for (int i =0; i < GEMM_BLOCK; i++) {
				scale_chunk[i] = scale_chunk[i] * current_scale;
			}

			__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
			for (int i =0; i < GEMM_BLOCK; i++) {
				current_row[col][i] = scale_chunk[i];
			}

			// Update local memory buffer with chunk
			__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
			for (int i =0; i < GEMM_BLOCK; i++) {
				a_buffer[k][col][kk][i] = scale_chunk[i];
			}
		}

		// Update all remaining rows
		#pragma loop_coalesce
		for (int row = 0; row < BLOCK_SIZE/GEMM_BLOCK - k; row++) {
			// Update whole rows!
			__attribute__((xcl_pipeline_loop(1)))
			for (int curr_col = 0; curr_col < BLOCK_SIZE/GEMM_BLOCK; curr_col++) {
				DEVICE_DATA_TYPE colbuf[GEMM_BLOCK];
				__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
				for (int j=0; j < GEMM_BLOCK; j++) {
					colbuf[j] = current_lu_col[row][j];
				}	
				__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
				for (int i = 0; i < GEMM_BLOCK; i++) {
					__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
					for (int j=0; j < GEMM_BLOCK; j++) {
						a_buffer[row + k][curr_col][i][j] += colbuf[i] * current_row[curr_col][j];
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
				__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
				for (int jj =0; jj < GEMM_BLOCK; jj++) {
					a[block_col * BLOCK_SIZE  + (block_row * BLOCK_SIZE + i * GEMM_BLOCK + ii) * BLOCK_SIZE * blocks_per_row + j * GEMM_BLOCK + jj] = a_buffer[i][j][ii][jj];
				}
			}
		}
	}
	// Store current block separately for easier transmission over host
	#pragma loop_coalesce
	for (int i =0; i < BLOCK_SIZE/GEMM_BLOCK; i++) {
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			for (int j =0; j < BLOCK_SIZE/GEMM_BLOCK; j++) {
				__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
				for (int jj =0; jj < GEMM_BLOCK; jj++) {
					top_block[(i * GEMM_BLOCK + ii) * BLOCK_SIZE + j * GEMM_BLOCK + jj] = a_buffer[i][j][ii][jj];
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
void left_update(__global DEVICE_DATA_TYPE* restrict a, 
				__global DEVICE_DATA_TYPE* restrict left_block,
				__global DEVICE_DATA_TYPE* restrict lu_global_buffer,
				const uint is_first_block,
				const uint block_col,
				const uint block_row,
				const uint blocks_per_row) {

	// Store current block in local memory
	local DEVICE_DATA_TYPE a_buffer[BLOCK_SIZE/GEMM_BLOCK][BLOCK_SIZE/GEMM_BLOCK][GEMM_BLOCK][GEMM_BLOCK] __attribute((xcl_array_partition(complete, 3),xcl_array_partition(complete, 4)));

	// Load block to local memory
	#pragma loop_coalesce
	for (int i =0; i < BLOCK_SIZE/GEMM_BLOCK; i++) {
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			for (int j =0; j < BLOCK_SIZE/GEMM_BLOCK; j++) {
				__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
				for (int jj =0; jj < GEMM_BLOCK; jj++) {
					a_buffer[i][j][ii][jj] = a[block_col * BLOCK_SIZE  + (block_row * BLOCK_SIZE + i * GEMM_BLOCK + ii) * BLOCK_SIZE * blocks_per_row + j * GEMM_BLOCK + jj];
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

		DEVICE_DATA_TYPE current_lu_row[BLOCK_SIZE/GEMM_BLOCK][GEMM_BLOCK] __attribute((xcl_array_partition(complete, 2)));
		DEVICE_DATA_TYPE current_col[BLOCK_SIZE/GEMM_BLOCK][GEMM_BLOCK] __attribute((xcl_array_partition(complete, 2)));

		for (int col = 0; col < BLOCK_SIZE / GEMM_BLOCK; col++) {
			DEVICE_DATA_TYPE chunk[GEMM_BLOCK];
			// get current row chunk
			__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
			for (int i =0; i < GEMM_BLOCK; i++) {
				chunk[i] = a_buffer[col][k][i][kk];
			}

			// Store chunk for later update
			ch_chunk_t col_out;
			__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
			for (int i =0; i < GEMM_BLOCK; i++) {
				current_col[col][i] = chunk[i];
			}

			ch_chunk_t row_in;
			
			// if current column data is still available read it in and store it in buffer
			if (col < BLOCK_SIZE / GEMM_BLOCK - k) {
				// Load LU data from global memory 
				__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
				for (int i=0; i < GEMM_BLOCK; i++) {
					row_in.data[i] = lu_global_buffer[gk * BLOCK_SIZE + (col + k) * GEMM_BLOCK + i];
				}
				__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
				for (int i =0; i < GEMM_BLOCK; i++) {
					current_lu_row[col][i] = (col > 0 || i > kk) ? row_in.data[i] : 0.f;
				}
			}
		}

		// Update all rows
		#pragma loop_coalesce
		// Update only remaining row chunks
		#pragma ivdep
		for (int curr_col = 0; curr_col < BLOCK_SIZE/GEMM_BLOCK - k; curr_col++) {
#ifdef INTEL_FPGA
			#pragma ivdep
#endif
#ifdef XILINX_FPGA
			__attribute__((xcl_pipeline_loop(1)))
#endif
			for (int row = 0; row < BLOCK_SIZE/GEMM_BLOCK; row++) {
				DEVICE_DATA_TYPE colbuf[GEMM_BLOCK];
				__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
				for (int j=0; j < GEMM_BLOCK; j++) {
					colbuf[j] = current_col[row][j];
				}	
				__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
				for (int i = 0; i < GEMM_BLOCK; i++) {
					__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
					for (int j=0; j < GEMM_BLOCK; j++) {
						a_buffer[row][curr_col + k][i][j] += current_lu_row[curr_col][j] * colbuf[i];
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
				__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
				for (int jj =0; jj < GEMM_BLOCK; jj++) {
					a[block_col * BLOCK_SIZE  + (block_row * BLOCK_SIZE + i * GEMM_BLOCK + ii) * BLOCK_SIZE * blocks_per_row + j * GEMM_BLOCK + jj] = a_buffer[i][j][ii][jj];
				}
			}
		}
	}

	// Store current block separately for easier transmission over host
	#pragma loop_coalesce
	for (int i =0; i < BLOCK_SIZE/GEMM_BLOCK; i++) {
		for (int ii =0; ii < GEMM_BLOCK; ii++) {
			for (int j =0; j < BLOCK_SIZE/GEMM_BLOCK; j++) {
				__attribute__((opencl_unroll_hint(GEMM_BLOCK)))
				for (int jj =0; jj < GEMM_BLOCK; jj++) {
					left_block[(i * GEMM_BLOCK + ii) * BLOCK_SIZE + j * GEMM_BLOCK + jj] = a_buffer[j][i][jj][ii];
				}
			}
		}
	}
}

// PY_CODE_GEN block_start [replace(local_variables=locals()) for i in range(num_replications)]

/**
Update the inner blocks using the left and right column and rows

 */
 __attribute__((uses_global_work_offset(0)))
__kernel
void inner_update_mm/*PY_CODE_GEN i*/(__global DEVICE_DATA_TYPE* restrict a, 
				__global DEVICE_DATA_TYPE* restrict left_global_buffer,
				__global DEVICE_DATA_TYPE* restrict top_global_buffer,
				const uint block_col,
				const uint block_row,
				const uint blocks_per_row) {

	// Store current block in local memory
	local DEVICE_DATA_TYPE a_buffer[BLOCK_SIZE/GEMM_BLOCK_MM][BLOCK_SIZE/GEMM_BLOCK_MM][GEMM_BLOCK_MM][GEMM_BLOCK_MM] __attribute((xcl_array_partition(complete, 3),xcl_array_partition(complete, 4)));
	local DEVICE_DATA_TYPE top_buffer[BLOCK_SIZE/GEMM_BLOCK_MM][BLOCK_SIZE/GEMM_BLOCK_MM][GEMM_BLOCK_MM][GEMM_BLOCK_MM] __attribute((xcl_array_partition(complete, 3),xcl_array_partition(complete, 4)));
	local DEVICE_DATA_TYPE left_buffer[BLOCK_SIZE/GEMM_BLOCK_MM][BLOCK_SIZE/GEMM_BLOCK_MM][GEMM_BLOCK_MM][GEMM_BLOCK_MM] __attribute((xcl_array_partition(complete, 3),xcl_array_partition(complete, 4)));


#ifdef INTEL_FPGA
	// If Intel FPGA, read all buffers in a single pipeline to reduce resource utilization
	#pragma loop_coalesce
	for (int i =0; i < BLOCK_SIZE/GEMM_BLOCK_MM; i++) {
		for (int ii =0; ii < GEMM_BLOCK_MM; ii++) {
			for (int j =0; j < BLOCK_SIZE/GEMM_BLOCK_MM; j++) {
				__attribute__((opencl_unroll_hint(GEMM_BLOCK_MM)))
				for (int jj =0; jj < GEMM_BLOCK_MM; jj++) {
					a_buffer[i][j][ii][jj] = a[block_col * BLOCK_SIZE  + (block_row * BLOCK_SIZE + i * GEMM_BLOCK_MM + ii) * BLOCK_SIZE * blocks_per_row + j * GEMM_BLOCK_MM + jj];
				}
				__attribute__((opencl_unroll_hint(GEMM_BLOCK_MM)))
				for (int jj =0; jj < GEMM_BLOCK_MM; jj++) {
					top_buffer[i][j][ii][jj] = top_global_buffer[(i * GEMM_BLOCK_MM + ii) * BLOCK_SIZE + j * GEMM_BLOCK_MM + jj];
				}
				__attribute__((opencl_unroll_hint(GEMM_BLOCK_MM)))
				for (int jj =0; jj < GEMM_BLOCK_MM; jj++) {
					left_buffer[i][j][ii][jj] = left_global_buffer[(i * GEMM_BLOCK_MM + ii) * BLOCK_SIZE + j * GEMM_BLOCK_MM + jj];
				}
			}
		}
	}
#endif
#ifdef XILINX_FPGA
	// If Xilinx FPGA, load blocks in separate pipelines to achieve memory bursts!

	// Load blocks to local memory
	__attribute__((xcl_pipeline_loop(1)))
	for (int i =0; i < BLOCK_SIZE/GEMM_BLOCK_MM; i++) {
		for (int ii =0; ii < GEMM_BLOCK_MM; ii++) {
			for (int j =0; j < BLOCK_SIZE/GEMM_BLOCK_MM; j++) {
				__attribute__((opencl_unroll_hint(GEMM_BLOCK_MM)))
				for (int jj =0; jj < GEMM_BLOCK_MM; jj++) {
					a_buffer[i][j][ii][jj] = a[block_col * BLOCK_SIZE  + (block_row * BLOCK_SIZE + i * GEMM_BLOCK_MM + ii) * BLOCK_SIZE * blocks_per_row + j * GEMM_BLOCK_MM + jj];
				}
			}
		}
	}

	__attribute__((xcl_pipeline_loop(1)))
	for (int i =0; i < BLOCK_SIZE/GEMM_BLOCK_MM; i++) {
		for (int ii =0; ii < GEMM_BLOCK_MM; ii++) {
			for (int j =0; j < BLOCK_SIZE/GEMM_BLOCK_MM; j++) {
				__attribute__((opencl_unroll_hint(GEMM_BLOCK_MM)))
				for (int jj =0; jj < GEMM_BLOCK_MM; jj++) {
					top_buffer[i][j][ii][jj] = top_global_buffer[(i * GEMM_BLOCK_MM + ii) * BLOCK_SIZE + j * GEMM_BLOCK_MM + jj];
				}
			}
		}
	}

	__attribute__((xcl_pipeline_loop(1)))
	for (int i =0; i < BLOCK_SIZE/GEMM_BLOCK_MM; i++) {
		for (int ii =0; ii < GEMM_BLOCK_MM; ii++) {
			for (int j =0; j < BLOCK_SIZE/GEMM_BLOCK_MM; j++) {
				__attribute__((opencl_unroll_hint(GEMM_BLOCK_MM)))
				for (int jj =0; jj < GEMM_BLOCK_MM; jj++) {
					left_buffer[i][j][ii][jj] = left_global_buffer[(i * GEMM_BLOCK_MM + ii) * BLOCK_SIZE + j * GEMM_BLOCK_MM + jj];
				}
			}
		}
	}
	#endif

	// Update whole block
	#pragma ivdep array(a_buffer) safelen((BLOCK_SIZE/GEMM_BLOCK_MM)*(BLOCK_SIZE/GEMM_BLOCK_MM))
	for (int c = 0; c < (BLOCK_SIZE/GEMM_BLOCK_MM) * (BLOCK_SIZE/GEMM_BLOCK_MM) * (BLOCK_SIZE/GEMM_BLOCK_MM); c++) {

		int mcol = c / ((BLOCK_SIZE/GEMM_BLOCK_MM)*(BLOCK_SIZE/GEMM_BLOCK_MM));
		int row = (c / (BLOCK_SIZE/GEMM_BLOCK_MM)) & ((BLOCK_SIZE/GEMM_BLOCK_MM) - 1);
		int curr_col = c & ((BLOCK_SIZE/GEMM_BLOCK_MM) - 1);

		DEVICE_DATA_TYPE top_sub[GEMM_BLOCK_MM][GEMM_BLOCK_MM];
		DEVICE_DATA_TYPE left_sub[GEMM_BLOCK_MM][GEMM_BLOCK_MM];

		__attribute__((opencl_unroll_hint(GEMM_BLOCK_MM)))
		for (int i = 0; i < GEMM_BLOCK_MM; i++) {
			__attribute__((opencl_unroll_hint(GEMM_BLOCK_MM)))
			for (int j=0; j < GEMM_BLOCK_MM; j++) {
				top_sub[i][j] = top_buffer[mcol][curr_col][i][j];
			}
		}

		__attribute__((opencl_unroll_hint(GEMM_BLOCK_MM)))
		for (int i = 0; i < GEMM_BLOCK_MM; i++) {
			__attribute__((opencl_unroll_hint(GEMM_BLOCK_MM)))
			for (int j=0; j < GEMM_BLOCK_MM; j++) {
				left_sub[i][j] = left_buffer[mcol][row][i][j];
			}
		}

		DEVICE_DATA_TYPE result_sub[GEMM_BLOCK_MM][GEMM_BLOCK_MM];
		__attribute__((opencl_unroll_hint(GEMM_BLOCK_MM)))
		for (int i = 0; i < GEMM_BLOCK_MM; i++) {
			__attribute__((opencl_unroll_hint(GEMM_BLOCK_MM)))
			for (int j = 0; j < GEMM_BLOCK_MM; j++) {
				// Calculate sum of whole column and only write it back once
				DEVICE_DATA_TYPE sum = 0.0;
				__attribute__((opencl_unroll_hint(GEMM_BLOCK_MM)))
				for (int k=0; k < GEMM_BLOCK_MM; k++) {
					sum += left_sub[k][i] * top_sub[k][j];
				}
				result_sub[i][j] = sum;
			}
		}

		__attribute__((opencl_unroll_hint(GEMM_BLOCK_MM)))
		for (int i = 0; i < GEMM_BLOCK_MM; i++) {
			__attribute__((opencl_unroll_hint(GEMM_BLOCK_MM)))
			for (int j=0; j < GEMM_BLOCK_MM; j++) {
				a_buffer[row][curr_col][i][j] += __fpga_reg(result_sub[i][j]);
			}
		}
	}

	// Store block to global memory
#ifdef INTELFPGA
	#pragma loop_coalesce
#endif
#ifdef XILINX_FPGA
	__attribute__((xcl_pipeline_loop(1)))
#endif
	for (int i =0; i < BLOCK_SIZE/GEMM_BLOCK_MM; i++) {
		for (int ii =0; ii < GEMM_BLOCK_MM; ii++) {
			for (int j =0; j < BLOCK_SIZE/GEMM_BLOCK_MM; j++) {
				__attribute__((opencl_unroll_hint(GEMM_BLOCK_MM)))
				for (int jj =0; jj < GEMM_BLOCK_MM; jj++) {
					a[block_col * BLOCK_SIZE  + (block_row * BLOCK_SIZE + i * GEMM_BLOCK_MM + ii) * BLOCK_SIZE * blocks_per_row + j * GEMM_BLOCK_MM + jj] = a_buffer[i][j][ii][jj];
				}
			}
		}
	}
}

// PY_CODE_GEN block_end
