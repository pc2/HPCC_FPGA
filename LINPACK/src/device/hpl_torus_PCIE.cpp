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

const unsigned block_size = (1 << LOCAL_MEM_BLOCK_LOG);
const unsigned gemm_block = (1 << REGISTER_BLOCK_LOG);
const unsigned gemm_block_mm = (1 << REGISTER_BLOCK_MM_LOG);

#ifdef KERNEL_lu
/**
Executes a single step of the LU factorization.

This method takes a partially solved 8x8 matrix and calculates the next step of
the LU factorization The method needs 7 (gemm_block-1) calls to perform a single
LU factorization. This is done to reduce resource usage, since all upcomng calls
are anyway depending on the results of the previous call and there is no way to
pipeline multiple executions.

A is the input block that might be partially computed
step is the current step and must be a value between 0 to gemm_block-2. After
step gemm_block-2, the block is factorized
 */
void lu_block(const DEVICE_DATA_TYPE A[gemm_block][gemm_block], const int step,
              DEVICE_DATA_TYPE A_out[gemm_block][gemm_block]) {

  // Read current line from input
  DEVICE_DATA_TYPE line[gemm_block];
  for (int i = 0; i < gemm_block; i++) {
    line[i] = A[step][i];
  }

  // calculate the inverse of the diagonal element for the scaling
  DEVICE_DATA_TYPE inv_scale_a = -1.0 / line[step];

  // Scale the current row
  for (int i = 0; i < gemm_block; i++) {
    if (i > step) {
      line[i] = line[i] * inv_scale_a;
    }
  }
  line[step] = inv_scale_a;

  // Update all rows fully unrolled
  // The multiply adds are fully independent
  //__attribute__((opencl_unroll_hint(gemm_block)))
  // Unrolling disabled for this loop to save resources
  for (int j = 0; j < gemm_block; j++) {
#pragma HLS PIPELINE II=1
    DEVICE_DATA_TYPE curr_scale = A[j][step];
    // Update a single row. If it is already updated, just write back the value,
    // if it is the current row write back the value in "line", else update the
    // value
    if (j != step) {
      for (int i = 0; i < gemm_block; i++) {
        A_out[j][i] =
            (i > step && j > step) ? A[j][i] + line[i] * curr_scale : A[j][i];
      }
    } else {
      for (int i = 0; i < gemm_block; i++) {
        A_out[j][i] = line[i];
      }
    }
  }
}

/**
This function can be used to update blocks using with three different
operations. It will execute the update for a single row in the block. The update
is completed after gemm_block calls of this update function

operation_type: 0 for top = the top row of blocks will need a triangular MM
                                1 for left = the left column of blocks will need
a triangular MM, matrices have to be transposed 2 for inner block == all inner
blocks will be updated with a MM
 */
void update_block(const DEVICE_DATA_TYPE a[gemm_block][gemm_block],
                  const DEVICE_DATA_TYPE top[gemm_block],
                  const DEVICE_DATA_TYPE left_or_lu[gemm_block],
                  DEVICE_DATA_TYPE out[gemm_block][gemm_block],
                  const int current_row, const int operation_type) {

  // Define different operation types of function
  const int op_top = 0;
  const int op_left = 1;
  const int op_inner = 2;

  // Transpose the input matrices if the target is a left block
  DEVICE_DATA_TYPE current_block[gemm_block][gemm_block];
  if (operation_type == op_left) {
    for (int ii = 0; ii < gemm_block; ii++) {
      for (int jj = 0; jj < gemm_block; jj++) {
        current_block[ii][jj] = a[jj][ii];
      }
    }
  } else {
    for (int ii = 0; ii < gemm_block; ii++) {
      for (int jj = 0; jj < gemm_block; jj++) {
        current_block[ii][jj] = a[ii][jj];
      }
    }
  }

  // Generate the first scalling array depending on the operation type
  DEVICE_DATA_TYPE scale_row[gemm_block];
  if (operation_type == op_inner) {
    for (int jj = 0; jj < gemm_block; jj++) {
      scale_row[jj] = top[jj];
    }
  } else {
    for (int jj = 0; jj < gemm_block; jj++) {
      scale_row[jj] = current_block[current_row][jj];
    }
  }
  if (operation_type == op_top) {
    for (int jj = 0; jj < gemm_block; jj++) {
      scale_row[jj] *= left_or_lu[current_row];
    }
  }

  DEVICE_DATA_TYPE tmp[gemm_block][gemm_block];
  // scale all values with the pre calculated scaling array and the second input
  for (int ii = 0; ii < gemm_block; ii++) {
    for (int jj = 0; jj < gemm_block; jj++) {
      // left_or_lu_block are stored transposed to simplify the data access here
      tmp[ii][jj] = current_block[ii][jj] + scale_row[jj] * left_or_lu[ii];
    }
  }

  // overwrite results that were calculated altough they are not needed for the
  // triangular operations left and top
  if (operation_type != op_inner) {
    for (int ii = 0; ii < gemm_block; ii++) {
      if (ii == current_row) {
        for (int jj = 0; jj < gemm_block; jj++) {
          tmp[ii][jj] = scale_row[jj];
        }
      } else if (ii < current_row) {
        for (int jj = 0; jj < gemm_block; jj++) {
          tmp[ii][jj] = current_block[ii][jj];
        }
      }
    }
  }

  // write result back and transpose if necessary
  if (operation_type == op_left) {
    for (int ii = 0; ii < gemm_block; ii++) {
      for (int jj = 0; jj < gemm_block; jj++) {
        out[ii][jj] = tmp[jj][ii];
      }
    }
  } else {
    for (int ii = 0; ii < gemm_block; ii++) {
      for (int jj = 0; jj < gemm_block; jj++) {
        out[ii][jj] = tmp[ii][jj];
      }
    }
  }
}

#endif

extern "C" {

#ifdef KERNEL_lu
void lu(DEVICE_DATA_TYPE *a, DEVICE_DATA_TYPE *a_block_trans,
        DEVICE_DATA_TYPE *a_block, const unsigned int block_col, const unsigned int block_row,
        const unsigned int blocks_per_row) {

  DEVICE_DATA_TYPE a_buffer[block_size / gemm_block][block_size / gemm_block]
                           [gemm_block][gemm_block];

  // Store current row and column in separate buffers for
  // easier access in the deep pipeline
  // need to be declared as local to prevent the compiler from
  DEVICE_DATA_TYPE top_buffer[block_size / gemm_block][gemm_block];
  DEVICE_DATA_TYPE left_buffer[block_size / gemm_block][gemm_block];

  // Load block to local memory
load_a_block:
  for (int i = 0; i < block_size / gemm_block; i++) {
    for (int ii = 0; ii < gemm_block; ii++) {
      for (int j = 0; j < block_size / gemm_block; j++) {
#pragma HLS PIPELINE
        for (int jj = 0; jj < gemm_block; jj++) {
          a_buffer[i][j][ii][jj] =
              a[block_col * block_size +
                (block_row * block_size + i * gemm_block + ii) * block_size *
                    blocks_per_row +
                j * gemm_block + jj];
        }
      }
    }
  }

  // For each row in the matrix update whole matrix.
  // The iterations depend on each other, so loop pipelining is disabled here
loop_diag:
  for (int gk = 0; gk < block_size; gk++) {

    int k = gk / gemm_block;
    int kk = gk & (gemm_block - 1);

    // Read in current LU block
    DEVICE_DATA_TYPE lu_a_buffer_in[gemm_block][gemm_block];
load_a_sb:
    for (int ii = 0; ii < gemm_block; ii++) {
      for (int jj = 0; jj < gemm_block; jj++) {
        lu_a_buffer_in[ii][jj] = a_buffer[k][k][ii][jj];
      }
    }

    DEVICE_DATA_TYPE lu_a_buffer_out[gemm_block][gemm_block];
    DEVICE_DATA_TYPE lu_a_buffer_out_row[gemm_block];
    DEVICE_DATA_TYPE lu_a_buffer_out_col[gemm_block];
    // Calculate next row and column of LU factorization and store in local
    // memory buffer
    lu_block(lu_a_buffer_in, kk, lu_a_buffer_out);
write_lu_sb:
    for (int ii = 0; ii < gemm_block; ii++) {
      for (int jj = 0; jj < gemm_block; jj++) {
        a_buffer[k][k][ii][jj] = lu_a_buffer_out[ii][jj];
      }
    }
write_lu_row:
    for (int jj = 0; jj < gemm_block; jj++) {
      lu_a_buffer_out_row[jj] = lu_a_buffer_out[kk][jj];
    }
write_lu_col:
    for (int jj = 0; jj < gemm_block; jj++) {
      lu_a_buffer_out_col[jj] = lu_a_buffer_out[jj][kk];
    }

    // The update pipeline does not need to be executed for the last
    // row of blocks
    if (gk < block_size - gemm_block) {

update_inner:
      // update all left blocks
      for (int tj = 1; tj < block_size / gemm_block; tj++) {
#pragma HLS PIPELINE II=1

        int j = k;
        int i = tj;

        if (i > k) {
          // copy the correct block in the second input buffer
          // this depends on the operations that has to be executed
          DEVICE_DATA_TYPE second_input[gemm_block];

          // left matrix block will be calculated
          for (int jj = 0; jj < gemm_block; jj++) {
            second_input[jj] = lu_a_buffer_out_row[jj];
          }
          DEVICE_DATA_TYPE a_input[gemm_block][gemm_block];
          for (int ii = 0; ii < gemm_block; ii++) {
            for (int jj = 0; jj < gemm_block; jj++) {
              a_input[ii][jj] = a_buffer[i][j][ii][jj];
            }
          }
          DEVICE_DATA_TYPE top_input[gemm_block];
          DEVICE_DATA_TYPE out[gemm_block][gemm_block];
          update_block(a_input, top_input, second_input, out, kk, 1);

          for (int ii = 0; ii < gemm_block; ii++) {
            left_buffer[i][ii] = out[ii][kk];
          }
          for (int ii = 0; ii < gemm_block; ii++) {
            for (int jj = 0; jj < gemm_block; jj++) {
              a_buffer[i][j][ii][jj] = out[ii][jj];
            }
          }
        }
      }

      // Update all other blocks with the new calculated row and column
      // First update top blocks, then update left blocks, then all inner blocks
      // ti == 0: top blocks
      // ti == 1: left blocks
      // ti > 1: inner blocks
update_inner_2:
      for (int ti = 0; ti < block_size / gemm_block - k; ti++) {
        for (int tj = 1; tj < block_size / gemm_block; tj++) {
#pragma HLS PIPELINE II=1

          int j = tj;
          int i = ti + k;
          // always execute the pipeline for whole rows of matrix blocks.
          // Only execute update for blocks that are required.
          // This helps to keep constant latencies between data dependencies of
          // the pipeline stages
          if ((i > k || ti == 0) && j > k) {

            // copy the correct block in the second input buffer
            // this depends on the operations that has to be executed
            DEVICE_DATA_TYPE second_input[gemm_block];
            if (ti == 0) {
              // top matrix block will be calculated
              for (int jj = 0; jj < gemm_block; jj++) {
                second_input[jj] = lu_a_buffer_out_col[jj];
              }
            } else {
              // inner block will be calculated
              for (int jj = 0; jj < gemm_block; jj++) {
                second_input[jj] = left_buffer[i][jj];
              }
            }
            DEVICE_DATA_TYPE a_input[gemm_block][gemm_block];
            for (int ii = 0; ii < gemm_block; ii++) {
              for (int jj = 0; jj < gemm_block; jj++) {
                a_input[ii][jj] = a_buffer[i][j][ii][jj];
              }
            }
            DEVICE_DATA_TYPE top_input[gemm_block];
            for (int jj = 0; jj < gemm_block; jj++) {
              top_input[jj] = top_buffer[j][jj];
            }
            DEVICE_DATA_TYPE out[gemm_block][gemm_block];
            update_block(a_input, top_input, second_input, out, kk,
                         (ti == 0) ? 0 : 2);
            if (ti == 0) {
              // only update in the first row
              for (int jj = 0; jj < gemm_block; jj++) {
                top_buffer[j][jj] = out[kk][jj];
              }
            }
            for (int ii = 0; ii < gemm_block; ii++) {
              for (int jj = 0; jj < gemm_block; jj++) {
                a_buffer[i][j][ii][jj] = out[ii][jj];
              }
            }
          }
        }
      }
    }
  }

  // Store block to global memory
store_a:
  for (int i = 0; i < block_size / gemm_block; i++) {
    for (int ii = 0; ii < gemm_block; ii++) {
      for (int j = 0; j < block_size / gemm_block; j++) {
        for (int jj = 0; jj < gemm_block; jj++) {
          a[block_col * block_size +
            (block_row * block_size + i * gemm_block + ii) * block_size *
                blocks_per_row +
            j * gemm_block + jj] = a_buffer[i][j][ii][jj];
        }
      }
    }
  }
  // Store current block in global memory also transposed to allow easier access
  // from the top kernel
  store_a_bt:
  for (int i = 0; i < block_size / gemm_block; i++) {
    for (int ii = 0; ii < gemm_block; ii++) {
      for (int j = 0; j < block_size / gemm_block; j++) {
        for (int jj = 0; jj < gemm_block; jj++) {
          a_block_trans[(i * gemm_block + ii) * block_size + j * gemm_block +
                        jj] = a_buffer[j][i][jj][ii];
        }
      }
    }
  }

store_a_b:
  for (int i = 0; i < block_size / gemm_block; i++) {
    for (int ii = 0; ii < gemm_block; ii++) {
      for (int j = 0; j < block_size / gemm_block; j++) {
        for (int jj = 0; jj < gemm_block; jj++) {
          a_block[(i * gemm_block + ii) * block_size + j * gemm_block + jj] =
              a_buffer[i][j][ii][jj];
        }
      }
    }
  }
}
#endif

#ifdef KERNEL_top_update
/**
Update the blocks to the right of the current LU block

 */
void top_update(DEVICE_DATA_TYPE *a, DEVICE_DATA_TYPE *top_block,
                const DEVICE_DATA_TYPE *lu_global_buffer_transposed,
                const unsigned int is_first_block, const unsigned int block_col,
                const unsigned int block_row, const unsigned int blocks_per_row) {

  // Store current block in local memory
  DEVICE_DATA_TYPE
      a_buffer[block_size / gemm_block][block_size / gemm_block][gemm_block]
              [gemm_block];

  // Load block to local memory
load_a:
  for (int i = 0; i < block_size / gemm_block; i++) {
    for (int ii = 0; ii < gemm_block; ii++) {
      for (int j = 0; j < block_size / gemm_block; j++) {
        for (int jj = 0; jj < gemm_block; jj++) {
          a_buffer[i][j][ii][jj] =
              a[block_col * block_size +
                (block_row * block_size + i * gemm_block + ii) * block_size *
                    blocks_per_row +
                j * gemm_block + jj];
        }
      }
    }
  }

// For each row in the matrix update whole matrix.
// The iterations depend on each other, so loop pipelining is disabled here
diag_exe:
  for (int gk = 0; gk < block_size; gk++) {

    int k = gk / gemm_block;
    int kk = gk & (gemm_block - 1);

    DEVICE_DATA_TYPE current_lu_col[block_size / gemm_block][gemm_block];
    DEVICE_DATA_TYPE current_row[block_size / gemm_block][gemm_block];
    DEVICE_DATA_TYPE current_scale;

scale_row:
    for (int col = 0; col < block_size / gemm_block; col++) {
#pragma HLS PIPELINE II=1
	    DEVICE_DATA_TYPE col_in[gemm_block];
#pragma HLS array_partition variable=col_in type=complete dim=0
      DEVICE_DATA_TYPE scale_chunk[gemm_block];
#pragma HLS array_partition variable=col_in type=complete dim=0

      // get current row chunk
      for (int i = 0; i < gemm_block; i++) {
        scale_chunk[i] = a_buffer[k][col][kk][i];
      }

      // if current column data is still available read it in and store it in
      // buffer
      if (col < block_size / gemm_block - k) {
        // Load LU data from global memory instead of receiving it from the
        // channel
        for (int i = 0; i < gemm_block; i++) {
          col_in[i] =
              lu_global_buffer_transposed[gk * block_size +
                                          (col + k) * gemm_block + i];
        }
        if (col == 0) {
          current_scale = col_in[kk];
        }
        for (int i = 0; i < gemm_block; i++) {
          current_lu_col[col][i] = (col > 0 || i > kk) ? col_in[i] : 0.f;
        }
      }

      // scale current row chunk with the rows scale factor received over the
      // external channel
      for (int i = 0; i < gemm_block; i++) {
        scale_chunk[i] = scale_chunk[i] * current_scale;
      }

      for (int i = 0; i < gemm_block; i++) {
        current_row[col][i] = scale_chunk[i];
      }

      // Update local memory buffer with chunk
      for (int i = 0; i < gemm_block; i++) {
        a_buffer[k][col][kk][i] = scale_chunk[i];
      }
    }

// Update all remaining rows
update_rows:
    for (int row = k; row < block_size / gemm_block; row++) {
#pragma HLS loop_tripcount min=0 max=block_size/gemm_block avg=block_size/gemm_block/2
      // Update whole rows!
      for (int curr_col = 0; curr_col < block_size / gemm_block; curr_col++) {
#pragma HLS PIPELINE II=1
        DEVICE_DATA_TYPE colbuf[gemm_block];
        for (int j = 0; j < gemm_block; j++) {
          colbuf[j] = current_lu_col[row - k][j];
        }
        for (int i = 0; i < gemm_block; i++) {
          for (int j = 0; j < gemm_block; j++) {
            a_buffer[row][curr_col][i][j] +=
                colbuf[i] * current_row[curr_col][j];
          }
        }
      }
    }
  }

// Store block to global memory
store_a:
  for (int i = 0; i < block_size / gemm_block; i++) {
    for (int ii = 0; ii < gemm_block; ii++) {
      for (int j = 0; j < block_size / gemm_block; j++) {
        for (int jj = 0; jj < gemm_block; jj++) {
          a[block_col * block_size +
            (block_row * block_size + i * gemm_block + ii) * block_size *
                blocks_per_row +
            j * gemm_block + jj] = a_buffer[i][j][ii][jj];
        }
      }
    }
  }
// Store current block separately for easier transmission over host
store_top:
  for (int i = 0; i < block_size / gemm_block; i++) {
    for (int ii = 0; ii < gemm_block; ii++) {
      for (int j = 0; j < block_size / gemm_block; j++) {
        for (int jj = 0; jj < gemm_block; jj++) {
          top_block[(i * gemm_block + ii) * block_size + j * gemm_block + jj] =
              a_buffer[i][j][ii][jj];
        }
      }
    }
  }
}
#endif

#ifdef KERNEL_left_update
/**
Update the blocks below the current LU block

 */
void left_update(DEVICE_DATA_TYPE * a,
                 DEVICE_DATA_TYPE * left_block,
                 const DEVICE_DATA_TYPE * lu_global_buffer,
                 const unsigned int is_first_block, const unsigned int block_col,
                 const unsigned int block_row, const unsigned int blocks_per_row) {

  // Store current block in local memory
  DEVICE_DATA_TYPE
      a_buffer[block_size / gemm_block][block_size / gemm_block][gemm_block]
              [gemm_block];

  // Load block to local memory
load_a:
  for (int i = 0; i < block_size / gemm_block; i++) {
    for (int ii = 0; ii < gemm_block; ii++) {
      for (int j = 0; j < block_size / gemm_block; j++) {
        for (int jj = 0; jj < gemm_block; jj++) {
          a_buffer[i][j][ii][jj] =
              a[block_col * block_size +
                (block_row * block_size + i * gemm_block + ii) * block_size *
                    blocks_per_row +
                j * gemm_block + jj];
        }
      }
    }
  }

  // For each row in the matrix update whole matrix.
  // The iterations depend on each other, so loop pipelining is disabled here
diag:
  for (int gk = 0; gk < block_size; gk++) {

    int k = gk / gemm_block;
    int kk = gk & (gemm_block - 1);

    DEVICE_DATA_TYPE current_lu_row[block_size / gemm_block][gemm_block];
    DEVICE_DATA_TYPE current_col[block_size / gemm_block][gemm_block];

first_col:
    for (int col = 0; col < block_size / gemm_block; col++) {
#pragma HLS PIPELINE II=1
      DEVICE_DATA_TYPE chunk[gemm_block];
      // get current row chunk
      for (int i = 0; i < gemm_block; i++) {
        chunk[i] = a_buffer[col][k][i][kk];
      }

      // Store chunk for later update
      for (int i = 0; i < gemm_block; i++) {
        current_col[col][i] = chunk[i];
      }

      DEVICE_DATA_TYPE row_in[gemm_block];

      // if current column data is still available read it in and store it in
      // buffer
      if (col < block_size / gemm_block - k) {
        // Load LU data from global memory
        for (int i = 0; i < gemm_block; i++) {
          row_in[i] =
              lu_global_buffer[gk * block_size + (col + k) * gemm_block + i];
        }
        for (int i = 0; i < gemm_block; i++) {
          current_lu_row[col][i] = (col > 0 || i > kk) ? row_in[i] : 0.f;
        }
      }
    }

    // Update all rows
    // Update only remaining row chunks
update:
    for (int curr_col = 0; curr_col < block_size / gemm_block - k; curr_col++) {
#pragma HLS loop_tripcount min=0 max=block_size/gemm_block avg=block_size/gemm_block/2
      for (int row = 0; row < block_size / gemm_block; row++) {
#pragma HLS PIPELINE II=1
        DEVICE_DATA_TYPE colbuf[gemm_block];
        for (int j = 0; j < gemm_block; j++) {
          colbuf[j] = current_col[row][j];
        }
        for (int i = 0; i < gemm_block; i++) {
          for (int j = 0; j < gemm_block; j++) {
            a_buffer[row][curr_col + k][i][j] +=
                current_lu_row[curr_col][j] * colbuf[i];
          }
        }
      }
    }
  }

  // Store block to global memory
store_a:
  for (int i = 0; i < block_size / gemm_block; i++) {
    for (int ii = 0; ii < gemm_block; ii++) {
      for (int j = 0; j < block_size / gemm_block; j++) {
        for (int jj = 0; jj < gemm_block; jj++) {
          a[block_col * block_size +
            (block_row * block_size + i * gemm_block + ii) * block_size *
                blocks_per_row +
            j * gemm_block + jj] = a_buffer[i][j][ii][jj];
        }
      }
    }
  }

  // Store current block separately for easier transmission over host
store_left:
  for (int i = 0; i < block_size / gemm_block; i++) {
    for (int ii = 0; ii < gemm_block; ii++) {
      for (int j = 0; j < block_size / gemm_block; j++) {
        for (int jj = 0; jj < gemm_block; jj++) {
          left_block[(i * gemm_block + ii) * block_size + j * gemm_block + jj] =
              a_buffer[j][i][jj][ii];
        }
      }
    }
  }
}
#endif

#ifdef KERNEL_inner_update_mm0
/**
Update the inner blocks using the left and right column and rows

 */
void inner_update_mm0(
    DEVICE_DATA_TYPE *a, const DEVICE_DATA_TYPE *left_global_buffer,
    const DEVICE_DATA_TYPE *top_global_buffer, const unsigned int block_col,
    const unsigned int block_row, const unsigned int blocks_per_row) {

  // Store current block in local memory
  DEVICE_DATA_TYPE a_buffer[block_size / gemm_block_mm]
                           [block_size / gemm_block_mm][gemm_block_mm]
                           [gemm_block_mm];
  DEVICE_DATA_TYPE top_buffer[block_size / gemm_block_mm]
                             [block_size / gemm_block_mm][gemm_block_mm]
                             [gemm_block_mm];
  DEVICE_DATA_TYPE left_buffer[block_size / gemm_block_mm]
                              [block_size / gemm_block_mm][gemm_block_mm]
                              [gemm_block_mm];

  // If Xilinx FPGA, load blocks in separate pipelines to achieve memory bursts!
  // Load blocks to local memory
load_a_block:
  for (int i = 0; i < block_size / gemm_block_mm; i++) {
    for (int ii = 0; ii < gemm_block_mm; ii++) {
      for (int j = 0; j < block_size / gemm_block_mm; j++) {
#pragma HLS PIPELINE II=1
        for (int jj = 0; jj < gemm_block_mm; jj++) {
          a_buffer[i][j][ii][jj] =
              a[block_col * block_size +
                (block_row * block_size + i * gemm_block_mm + ii) * block_size *
                    blocks_per_row +
                j * gemm_block_mm + jj];
        }
      }
    }
  }

load_top_block:
  for (int i = 0; i < block_size / gemm_block_mm; i++) {
    for (int ii = 0; ii < gemm_block_mm; ii++) {
      for (int j = 0; j < block_size / gemm_block_mm; j++) {
#pragma HLS PIPELINE II=1
        for (int jj = 0; jj < gemm_block_mm; jj++) {
          top_buffer[i][j][ii][jj] =
              top_global_buffer[(i * gemm_block_mm + ii) * block_size +
                                j * gemm_block_mm + jj];
        }
      }
    }
  }

load_left_block:
  for (int i = 0; i < block_size / gemm_block_mm; i++) {
    for (int ii = 0; ii < gemm_block_mm; ii++) {
      for (int j = 0; j < block_size / gemm_block_mm; j++) {
#pragma HLS PIPELINE II=1
        for (int jj = 0; jj < gemm_block_mm; jj++) {
          left_buffer[i][j][ii][jj] =
              left_global_buffer[(i * gemm_block_mm + ii) * block_size +
                                 j * gemm_block_mm + jj];
        }
      }
    }
  }

  // Update whole block
calc_subblocks:
  for (int c = 0;
       c < (block_size / gemm_block_mm) * (block_size / gemm_block_mm) *
               (block_size / gemm_block_mm);
       c++) {
#pragma HLS PIPELINE II=1

    int mcol =
        c / ((block_size / gemm_block_mm) * (block_size / gemm_block_mm));
    int row =
        (c / (block_size / gemm_block_mm)) % (block_size / gemm_block_mm);
    int curr_col = c & ((block_size / gemm_block_mm) - 1);

    DEVICE_DATA_TYPE top_sub[gemm_block_mm][gemm_block_mm];
    DEVICE_DATA_TYPE left_sub[gemm_block_mm][gemm_block_mm];

load_top_sb:
    for (int i = 0; i < gemm_block_mm; i++) {
      for (int j = 0; j < gemm_block_mm; j++) {
        top_sub[i][j] = top_buffer[mcol][curr_col][i][j];
      }
    }

load_left_sb:
    for (int i = 0; i < gemm_block_mm; i++) {
      for (int j = 0; j < gemm_block_mm; j++) {
        left_sub[i][j] = left_buffer[mcol][row][i][j];
      }
    }

    DEVICE_DATA_TYPE result_sub[gemm_block_mm][gemm_block_mm];
mmul:
    for (int i = 0; i < gemm_block_mm; i++) {
      for (int j = 0; j < gemm_block_mm; j++) {
        // Calculate sum of whole column and only write it back once
        DEVICE_DATA_TYPE sum = 0.0;
        for (int k = 0; k < gemm_block_mm; k++) {
          sum += left_sub[k][i] * top_sub[k][j];
        }
        result_sub[i][j] = sum;
      }
    }

add_sb:
    for (int i = 0; i < gemm_block_mm; i++) {
      for (int j = 0; j < gemm_block_mm; j++) {
        a_buffer[row][curr_col][i][j] += result_sub[i][j];
      }
    }
  }

  // Store block to global memory
store_result:
  for (int i = 0; i < block_size / gemm_block_mm; i++) {
    for (int ii = 0; ii < gemm_block_mm; ii++) {
      for (int j = 0; j < block_size / gemm_block_mm; j++) {
        for (int jj = 0; jj < gemm_block_mm; jj++) {
          a[block_col * block_size +
            (block_row * block_size + i * gemm_block_mm + ii) * block_size *
                blocks_per_row +
            j * gemm_block_mm + jj] = a_buffer[i][j][ii][jj];
        }
      }
    }
  }
}

#endif
}
